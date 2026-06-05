#include "common.h"
#include "GaussianBlurFilter.h"

template<int Channels>
void GaussianBlurFilter<Channels>::apply(const Image<Channels> &image) const {
    auto constexpr bytesPerPixel = sizeof(Npp32f) * Channels;

    // Get image dimensions
    auto const size = image.getSize();
    size_t const nSrcStep = size.width * bytesPerPixel;

    // Get CUDA stream context
    auto const nppStreamCtx = &this->channel_->getNppStreamContext();
    auto const cudaStream = nppStreamCtx->hStream;

    size_t tmpDstPitch;
    Npp32f *deviceResultPtr = nullptr;
    if (cudaMallocPitch(&deviceResultPtr, &tmpDstPitch, nSrcStep, size.height) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for Gaussian blur result");
    }

    // Fetch the device ptr
    auto const deviceSrcPtr = static_cast<Npp32f *>(this->channel_->getDevicePtr());
    if (!deviceSrcPtr) {
        throw std::invalid_argument("Device pointer is null");
    }

    // Apply the Gaussian blur
    NppStatus status;
    switch (Channels) {
        case 1:
            status = nppiFilterGauss_32f_C1R_Ctx(
                deviceSrcPtr, nSrcStep, deviceResultPtr, nSrcStep, size, kernelSize_, *nppStreamCtx);
            break;
        case 3:
            status = nppiFilterGauss_32f_C3R_Ctx(
                deviceSrcPtr, nSrcStep, deviceResultPtr, nSrcStep, size, kernelSize_, *nppStreamCtx);
            break;
        case 4:
            status = nppiFilterGauss_32f_C4R_Ctx(
                deviceSrcPtr, nSrcStep, deviceResultPtr, nSrcStep, size, kernelSize_, *nppStreamCtx);
            break;
        default:
            throw std::invalid_argument("Unsupported number of channels");
    }

    if (status != NPP_SUCCESS) {
        throw std::runtime_error("NPP Gaussian blur operation failed with status: " + std::to_string(status));
    }

    if (cudaMemcpy2DAsync(deviceSrcPtr, nSrcStep, deviceResultPtr, tmpDstPitch, nSrcStep, size.height,
                          cudaMemcpyDeviceToDevice, cudaStream) != cudaSuccess) {
        throw std::runtime_error("CUDA memory copy operation failed");
    }

    cudaFreeAsync(deviceResultPtr, cudaStream);
}

template<int Channels>
GaussianBlurFilter<
    Channels>::GaussianBlurFilter(CudaChannel &channel,
                                  const NppiMaskSize kernelSize) noexcept : ICudaImageFilter<Channels>(
                                                                                channel),
                                                                            kernelSize_(kernelSize) {
}

template class GaussianBlurFilter<1>;

template class GaussianBlurFilter<3>;

template class GaussianBlurFilter<4>;
