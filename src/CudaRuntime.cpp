#include "CudaRuntime.h"

#include <cuda.h>
#include <stdexcept>

CudaRuntime::CudaRuntime() : event(nullptr), stream(nullptr) {
    cuInit(0);
    device_prop = std::make_unique<cudaDeviceProp>();
    if (cudaGetDeviceProperties(device_prop.get(), 0) != cudaSuccess) {
        throw std::runtime_error("Failed to get CUDA device properties");
    }

    stream = std::make_unique<cudaStream_t>();
    event = std::make_unique<cudaEvent_t>();
    cudaSetDevice(0);

    if (cudaStreamCreate(stream.get()) != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream");
    }
    if (cudaEventCreate(event.get()) != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA event");
    }
}

std::string CudaRuntime::getDeviceName() const {
    return device_prop->name;
}

void CudaRuntime::synchronize() const {
    if (cudaStreamSynchronize(*stream) != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize CUDA stream");
    }
}

CudaRuntime::~CudaRuntime() {
    cudaEventDestroy(*event);
    if (auto const ptr = stream.release()) {
        cudaStreamDestroy(*ptr);
    }
    cuDevicePrimaryCtxRelease(0);
}
