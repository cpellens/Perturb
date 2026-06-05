#include "NormalMapFilter.h"
#include "CudaError.h"

#include "normal_map.cu"

template<int Channels>
NormalMapFilter<
    Channels>::NormalMapFilter(CudaChannel &channel, NppiMaskSize blurKernelSize) : ICudaImageFilter<Channels>(channel),
    sobelFilter(channel),
    gaussianBlurFilter(channel, blurKernelSize),
    grayscaleFilter(channel) {
}

template<int Channels>
void NormalMapFilter<Channels>::apply(const Image<1> &a, const Image<1> &b) const {
    auto constexpr bytesPerPixel = sizeof(Npp32f);

    const auto [width, height] = a.getSize();
    auto const nSrcStep = width * bytesPerPixel;

    // Fetch CUDA contexts
    auto const nppStreamCtx = &this->channel_->getNppStreamContext();
    if (nppStreamCtx == nullptr) {
        throw std::runtime_error("NPP stream context is null, cannot execute normal map filter");
    }
    auto const cudaStream = nppStreamCtx->hStream;
    if (cudaStream == nullptr) {
        throw std::runtime_error("CUDA stream is null, cannot execute normal map filter");
    }

    const auto devicePtrA = a.createDevicePtr(*this->channel_);
    const auto devicePtrB = b.createDevicePtr(*this->channel_);

    // TODO: Merge the two grayscale images into a single two-channel image?

    // Now we just need one more pointer to store the result. We'll store it in the channel.
    float3 *dstPtr = nullptr;
    if (auto const statusAlloc = cudaMallocAsync(
        &dstPtr,
        sizeof(float3) * width * height,
        cudaStream); statusAlloc != cudaSuccess) {
        handleCudaError(statusAlloc);
    }

    cudaStreamSynchronize(cudaStream);

    constexpr dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Run the normal map kernel
    launchBuildNormalMap(gridSize, blockSize, devicePtrA, devicePtrB, dstPtr, nSrcStep, width, height, 1);
    cudaDeviceSynchronize();

    // Store the result in the channel.
    // The output will always be three channels here.
    this->channel_->setDevicePtr(dstPtr, nSrcStep * height * 3);
}

template<int Channels>
void NormalMapFilter<Channels>::apply(const Image<Channels> &image) const {
    auto const [width, height] = image.getSize();

    Image<Channels> sobelResultX{width, height};
    sobelResultX.readFromChannel(*this->channel_);

    // Sync
    auto nppStreamCtx = &this->channel_->getNppStreamContext();
    cudaStreamSynchronize(nppStreamCtx->hStream);

    // Create a copy of the above result
    Image<Channels> sobelResultY{sobelResultX};

    // Perform Sobel filter on the horizontal axis
    sobelFilter.apply(sobelResultX, SobelFilter<Channels>::Horizontal);
    gaussianBlurFilter.apply(sobelResultX);
    grayscaleFilter.apply(sobelResultX);

    Image<1> sobelResultXGrayscale{width, height};
    sobelResultXGrayscale.readFromChannel(*this->channel_);
    cudaStreamSynchronize(nppStreamCtx->hStream);

    // Do the same on the vertical axis
    // First, need to write the result image to the channel
    sobelResultY.writeToChannel(*this->channel_);
    cudaStreamSynchronize(nppStreamCtx->hStream);

    // Perform Sobel filter on the vertical axis
    sobelFilter.apply(sobelResultY, SobelFilter<Channels>::Vertical);
    gaussianBlurFilter.apply(sobelResultY);
    grayscaleFilter.apply(sobelResultY);

    Image<1> sobelResultYGrayscale{width, height};
    sobelResultYGrayscale.readFromChannel(*this->channel_);

    // Now we can apply the normal map calculation
    apply(sobelResultXGrayscale, sobelResultYGrayscale);
}

template class NormalMapFilter<3>;

template class NormalMapFilter<4>;
