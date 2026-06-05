#include "common.h"
#include "SobelFilter.h"
#include "CudaError.h"

#define CALL_FILTER(channels, direction) \
    nppiFilterSobel##direction##Border_32f_C##channels##R_Ctx(imageDevicePtr, \
                                                tmpDstPitch, size, offset, \
                                                deviceResultPtr, \
                                                tmpDstPitch, size, \
                                                borderType, nppStreamCtx);

SobelFilter::SobelFilter(CudaChannel &channel) : channel_(channel) {
}

template<int Channels>
void SobelFilter::apply(const Image<Channels> &image, const Direction direction) const {
    // Allocate space for gradient images
    auto constexpr channels = Image<Channels>::getChannels();
    auto constexpr borderType = NPP_BORDER_REPLICATE;
    auto constexpr offset = NppiPoint{0, 0};

    // Get image dimensions
    auto const size = image.getSize();
    auto const width = size.width;
    auto const height = size.height;

    // Fetch CUDA contexts
    auto const nppStreamCtx = channel_.getNppStreamContext();
    auto const cudaStream = nppStreamCtx.hStream;
    auto const nSrcStep = width * channels * sizeof(Npp32f);

    // Create a pointer for the result of the gradient image
    size_t tmpDstPitch;
    Npp32f *deviceResultPtr = nullptr;
    if (cudaMallocPitch(&deviceResultPtr, &tmpDstPitch, nSrcStep, height) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for Sobel filter result");
    }

    // This should now be the pointer to the image data on the GPU
    const auto imageDevicePtr = static_cast<Npp32f *>(channel_.getDevicePtr());

    // Perform Sobel filter operation
    NppStatus status = NPP_SUCCESS;
    switch (Channels) {
        case 1:
            if (direction == Horizontal) {
                status = CALL_FILTER(1, Horiz);
            } else if (direction == Vertical) {
                status = CALL_FILTER(1, Vert);
            } else {
                throw std::invalid_argument("Unsupported direction");
            }
            break;
        case 3:
            if (direction == Horizontal) {
                status = CALL_FILTER(3, Horiz);
            } else if (direction == Vertical) {
                status = CALL_FILTER(3, Vert);
            } else {
                throw std::invalid_argument("Unsupported direction");
            }
            break;
        case 4:
            if (direction == Horizontal) {
                status = CALL_FILTER(4, Horiz);
            } else if (direction == Vertical) {
                status = CALL_FILTER(4, Vert);
            } else {
                throw std::invalid_argument("Unsupported direction");
            }
            break;
        default:
            throw std::invalid_argument("Unsupported number of channels");
    }

    // Check for errors
    if (status != NPP_SUCCESS) {
        handleNppError(status);
    }

    // Write the result back to the channel
    if (auto copyStatus = cudaMemcpy2DAsync(imageDevicePtr, nSrcStep, deviceResultPtr, tmpDstPitch, nSrcStep,
                                            size.height, cudaMemcpyDeviceToDevice,
                                            cudaStream); copyStatus != cudaSuccess) {
        handleCudaError(copyStatus);
    }

    // Free the result pointer once complete
    cudaFreeAsync(deviceResultPtr, cudaStream);
}

template void SobelFilter::apply<1>(const Image<1> &image, Direction direction) const;

template void SobelFilter::apply<3>(const Image<3> &image, Direction direction) const;

template void SobelFilter::apply<4>(const Image<4> &image, Direction direction) const;
