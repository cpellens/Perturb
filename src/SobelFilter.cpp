#include "common.h"
#include "SobelFilter.h"
#include "CudaError.h"

#define CALL_FILTER(channels, direction) \
    nppiFilterSobel##direction##Border_32f_C##channels##R_Ctx(imageDevicePtr, \
                                                tmpDstPitch, size, offset, \
                                                deviceResultPtr, \
                                                tmpDstPitch, size, \
                                                borderType, *nppStreamCtx);

template<int Channels>
void SobelFilter<Channels>::apply(const Image<Channels> &image, const Direction direction) const {
    // Allocate space for gradient images
    auto constexpr borderType = NPP_BORDER_REPLICATE;
    auto constexpr offset = NppiPoint{0, 0};

    // Get image dimensions
    auto const size = image.getSize();
    auto const width = size.width;
    auto const height = size.height;

    // Fetch CUDA contexts
    auto const nppStreamCtx = &this->channel_->getNppStreamContext();
    auto const nSrcStep = width * Channels * sizeof(Npp32f);

    // Create a pointer for the result of the gradient image
    size_t tmpDstPitch;
    Npp32f *deviceResultPtr = nullptr;
    if (cudaMallocPitch(&deviceResultPtr, &tmpDstPitch, nSrcStep, height) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for Sobel filter result");
    }

    // This should now be the pointer to the image data on the GPU
    const auto imageDevicePtr = static_cast<Npp32f *>(this->channel_->getDevicePtr());

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

    // Frees the previous memory then updates the internal pointer
    this->channel_->setDevicePtr(deviceResultPtr, tmpDstPitch * height);
}

template<int Channels>
void SobelFilter<Channels>::apply(const Image<Channels> &image) const {
    apply(image, Direction::Horizontal);
    apply(image, Direction::Vertical);
}

template class SobelFilter<1>;
template class SobelFilter<3>;
template class SobelFilter<4>;
