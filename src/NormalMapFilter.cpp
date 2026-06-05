#include "NormalMapFilter.h"
#include "CudaError.h"

#include "normal_map.cu"

void NormalMapFilter::apply(const Image<1> &a, const Image<1> &b) const {
    auto constexpr bytesPerPixel = sizeof(Npp32f);

    const auto [width, height] = a.getSize();
    auto const nSrcStep = width * bytesPerPixel;

    // Fetch CUDA contexts
    auto const nppStreamCtx = &channel_.getNppStreamContext();
    if (nppStreamCtx == nullptr) {
        throw std::runtime_error("NPP stream context is null, cannot execute normal map filter");
    }
    auto const cudaStream = nppStreamCtx->hStream;
    if (cudaStream == nullptr) {
        throw std::runtime_error("CUDA stream is null, cannot execute normal map filter");
    }

    const auto devicePtrA = a.createDevicePtr(channel_);
    const auto devicePtrB = b.createDevicePtr(channel_);

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
    channel_.setDevicePtr(dstPtr, nSrcStep * height * 3);
}
