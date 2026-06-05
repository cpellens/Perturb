#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include <string>

class CudaRuntime {
    std::unique_ptr<cudaEvent_t> event;
    std::unique_ptr<cudaDeviceProp> device_prop;

protected:
    friend class CudaChannel;
    std::unique_ptr<cudaStream_t> stream;

public:
    CudaRuntime();

    ~CudaRuntime();

    std::string getDeviceName() const;

    void synchronize() const;
};
