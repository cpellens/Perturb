#include "GrayscaleFilter.h"
#include "CudaError.h"

constexpr Npp32f grayscaleCoefficients[3] = {0.2126f, 0.7152f, 0.0722f};

template<int Channels>
void GrayscaleFilter<Channels>::apply(const Image<Channels> &image) const {
    auto constexpr srcBytesPerPixel = sizeof(Npp32f) * Channels;
    const auto [width, height] = image.getSize();

    // Step sizing
    auto const nSrcStep = width * srcBytesPerPixel;
    auto const nDstStep = width * sizeof(Npp32f);

    auto const nppStreamCtx = &this->channel_->getNppStreamContext();
    if (!nppStreamCtx) {
        throw std::invalid_argument("NPP stream context is null");
    }

    auto const cudaStream = nppStreamCtx->hStream;
    if (!cudaStream) {
        throw std::invalid_argument("CUDA stream is null");
    }

    // Create output pointer
    Npp32f *pDst = nullptr;
    if (auto const allocResult = cudaMallocAsync(&pDst, width * sizeof(Npp32f) * height, cudaStream);
        allocResult != cudaSuccess) {
        handleCudaError(allocResult);
    }

    // Create a read pointer
    auto const pSrc = static_cast<Npp32f *>(this->channel_->getDevicePtr());
    if (!pSrc) {
        throw std::invalid_argument("Device pointer is null");
    }

    NppStatus status;
    switch (Channels) {
        case 3:
            status = nppiColorToGray_32f_C3C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, {width, height},
                                                   grayscaleCoefficients,
                                                   *nppStreamCtx);
            break;
        case 4:
            status = nppiColorToGray_32f_C4C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, {width, height},
                                                   grayscaleCoefficients,
                                                   *nppStreamCtx);
            break;
        default:
            throw std::invalid_argument("Unsupported number of channels");
    }

    // Check for errors
    if (status != NPP_SUCCESS) {
        cudaFreeAsync(pDst, cudaStream);
        handleNppError(status);
    }

    // Replace it with the new pointer
    this->channel_->setDevicePtr(pDst, nDstStep * height);
}

template class GrayscaleFilter<3>;

template class GrayscaleFilter<4>;
