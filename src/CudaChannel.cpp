#include "CudaChannel.h"

#include <iostream>
#include <stdexcept>

void CudaChannel::freeDevicePtr() noexcept {
    cudaFreeAsync(*device_ptr, npp_stream_context->hStream);
    device_ptr.reset(nullptr);
}

void CudaChannel::createDevicePtr(const size_t size) {
    if (!npp_stream_context->hStream) {
        throw std::runtime_error("CUDA stream is not initialized");
    }

    if (device_ptr) {
        freeDevicePtr();
    }

    device_ptr = std::make_unique<void *>();
    if (const auto allocStatus = cudaMallocAsync(&*device_ptr, size, npp_stream_context->hStream);
        allocStatus != cudaSuccess) {
        throw std::runtime_error(
            "Failed to allocate device memory: " + std::string(cudaGetErrorString(allocStatus)));
    }
    allocation_size = size;

#ifdef _DEBUG
    std::cout << "[CudaChannel] Allocated " << size << " bytes on device" << std::endl;
#endif
}

void CudaChannel::createDevicePtr(const size_t size, const void *data) {
    createDevicePtr(size);
    if (data) {
        write(size, data);
    }
}

void CudaChannel::setDevicePtr(void *ptr, const size_t size) {
    if (!ptr) {
        throw std::invalid_argument("Device pointer cannot be null");
    }

    if (device_ptr) {
        freeDevicePtr();
        allocation_size = 0;
        device_ptr.reset(nullptr);
        cudaStreamSynchronize(npp_stream_context->hStream);
    }

    device_ptr = std::make_unique<void *>(ptr);
    allocation_size = size;
}

const NppStreamContext &CudaChannel::getNppStreamContext() const noexcept {
    return *npp_stream_context;
}

CudaChannel::CudaChannel(const CudaRuntime &runtime) {
    auto const deviceProps = runtime.device_prop.get();

    // Initialize NPP context
    npp_stream_context = std::make_unique<NppStreamContext>();
    npp_stream_context->hStream = *runtime.stream;
    npp_stream_context->nCudaDeviceId = deviceProps->deviceNumaId;
    npp_stream_context->nStreamFlags = 0;
    npp_stream_context->nMultiProcessorCount = deviceProps->multiProcessorCount;
    npp_stream_context->nMaxThreadsPerBlock = deviceProps->maxThreadsPerBlock;
    npp_stream_context->nMaxThreadsPerMultiProcessor = deviceProps->maxThreadsPerMultiProcessor;
    npp_stream_context->nSharedMemPerBlock = deviceProps->sharedMemPerBlock;
    npp_stream_context->nCudaDevAttrComputeCapabilityMajor = deviceProps->major;
    npp_stream_context->nCudaDevAttrComputeCapabilityMinor = deviceProps->minor;
    npp_stream_context->nReserved0 = 0;

    device_ptr = std::make_unique<void *>(nullptr);
    allocation_size = 0;
}

CudaChannel::~CudaChannel() {
    if (device_ptr) {
        freeDevicePtr();
    }
    npp_stream_context.reset();
    device_ptr.reset();
    allocation_size = 0;
}

void *CudaChannel::getDevicePtr() const noexcept {
    return *device_ptr;
}

void *CudaChannel::read(const size_t size) const {
    void *dst = std::malloc(size);
    if (!dst) {
        throw std::runtime_error("Failed to allocate host memory for reading from device");
    }
    read(size, dst);
    return dst;
}

void CudaChannel::read(const size_t size, void *dst) const {
    if (!dst) {
        throw std::invalid_argument("Destination pointer cannot be null");
    }

    if (!npp_stream_context->hStream) {
        throw std::runtime_error("CUDA stream is not initialized");
    }

    if (!device_ptr) {
        throw std::runtime_error("Device pointer is not initialized");
    }

    if (allocation_size < size) {
        throw std::runtime_error("Device memory allocation size is smaller than the requested size");
    }

    // Copy data from device to host
    if (const auto copyStatus = cudaMemcpyAsync(dst, *device_ptr, size, cudaMemcpyDeviceToHost,
                                                npp_stream_context->hStream);
        copyStatus != cudaSuccess) {
        throw std::runtime_error(
            "Failed to copy data from device to host: " + std::string(cudaGetErrorString(copyStatus)));
    }
}

void CudaChannel::write(const size_t size, const void *src) {
    if (!src) {
        throw std::invalid_argument("Data pointer cannot be null");
    }

    // Allocate device memory if needed
    if (allocation_size < size) {
        createDevicePtr(size);
    }

    // Copy data from host to device
    if (const auto copyStatus = cudaMemcpyAsync(*device_ptr, src, size, cudaMemcpyHostToDevice,
                                                npp_stream_context->hStream);
        copyStatus != cudaSuccess) {
        throw std::runtime_error(
            "Failed to copy data from host to device: " + std::string(cudaGetErrorString(copyStatus)));
    }
}
