#pragma once

#include <memory>
#include <nppdefs.h>

#include "CudaRuntime.h"

/**
 * @brief CudaChannel class provides a simplified interface for synchronizing
 * image data to and from the GPU.
 */
class CudaChannel {
    std::unique_ptr<void *> device_ptr;
    std::unique_ptr<NppStreamContext> npp_stream_context;
    size_t allocation_size;

public:
    explicit CudaChannel(const CudaRuntime &runtime);

    explicit CudaChannel(const CudaRuntime *runtime) : CudaChannel(*runtime) {
    }

    ~CudaChannel();

    [[nodiscard]] void *getDevicePtr() const noexcept;

    [[nodiscard]] void *read(size_t size) const;

    void read(size_t size, void *dst) const;

    void write(size_t size, const void *src);

    void freeDevicePtr() noexcept;

    void createDevicePtr(size_t size);

    void createDevicePtr(size_t size, const void *data);

    void setDevicePtr(void *ptr, size_t size);

    const NppStreamContext &getNppStreamContext() const noexcept;
};
