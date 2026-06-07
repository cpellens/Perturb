#include "NormalMapFilter.h"
#include "CudaError.h"

#include "normal_map.cu"

template<int Channels>
NormalMapFilter<Channels>::NormalMapFilter(
    CudaChannel &channel,
    const NppiMaskSize blurKernelSize)
    : ICudaImageFilter<Channels>(channel),
      sobelFilter(channel),
      gaussianBlurFilter(channel, blurKernelSize),
      grayscaleFilter(channel) {
}

template<int Channels>
void NormalMapFilter<Channels>::apply(const Image<1> &a, const Image<1> &b) const {
    auto constexpr bytesPerPixelChannel = sizeof(Npp32f);

    const auto [width, height] = a.getSize();
    auto const nSrcStep = width * bytesPerPixelChannel;

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

    // Calculate the grid size
    constexpr dim3 blockSize(16, 16);

    auto const gridSizeX = (width + blockSize.x - 1) / blockSize.x;
    auto const gridSizeY = (height + blockSize.y - 1) / blockSize.y;
    const dim3 gridSize(gridSizeX, gridSizeY);

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

    // Fetch the NPP context
    auto const nppStreamCtx = this->getNppStreamContext();
    auto const cudaStream = nppStreamCtx->hStream;

    // First, we can perform the grayscale filter on the input image to avoid
    // processing the Sobel filters 3x per axis
    grayscaleFilter.apply(image);

    Image<1> sobelResultX{width, height};
    sobelResultX.readFromChannel(this->channel_);

    // Create a copy of the above result
    Image sobelResultY{sobelResultX};

    // Perform Sobel filter on the horizontal axis
    sobelFilter.apply(sobelResultX, SobelFilter<1>::Horizontal);
    gaussianBlurFilter.apply(sobelResultX);

    // Read horizontal result into host memory
    sobelResultX.readFromChannel(this->channel_);
    cudaStreamSynchronize(cudaStream);

    // Do the same on the vertical axis
    sobelResultY.writeToChannel(this->channel_);

    // Perform Sobel filter on the vertical axis
    sobelFilter.apply(sobelResultY, SobelFilter<1>::Vertical);
    gaussianBlurFilter.apply(sobelResultY);

    sobelResultY.readFromChannel(this->channel_);

    // Now we can apply the normal map calculation
    apply(sobelResultX, sobelResultY);
}

template<int Channels>
const NppStreamContext *ICudaImageFilter<Channels>::getNppStreamContext() const {
    return &channel_->getNppStreamContext();
}

template class NormalMapFilter<3>;

template class NormalMapFilter<4>;
