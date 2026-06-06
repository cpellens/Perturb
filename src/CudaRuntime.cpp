#include "CudaRuntime.h"

#include <cuda.h>
#include <stdexcept>

#include "CudaError.h"

CudaRuntime::CudaRuntime() : stream(nullptr) {
    cuInit(0);
    device_prop = std::make_unique<cudaDeviceProp>();
    if (cudaGetDeviceProperties(device_prop.get(), 0) != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device properties");
    }

    stream = std::make_unique<cudaStream_t>();
    cudaSetDevice(0);

    if (cudaStreamCreate(stream.get()) != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }
}

std::string CudaRuntime::getDeviceName() const {
    return device_prop->name;
}

void CudaRuntime::synchronize() const {
    if (auto const status = cudaStreamSynchronize(*stream); status != cudaSuccess) {
        handleCudaError(status);
    }
}

CudaRuntime::~CudaRuntime() {
    if (auto const ptr = stream.release()) {
        cudaStreamDestroy(*ptr);
    }
    cudaDeviceReset();
}
