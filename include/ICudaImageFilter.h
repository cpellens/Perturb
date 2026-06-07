#pragma once

#include "Image.h"
#include "CudaChannel.h"

template<int Channels>
class ICudaImageFilter {
public:
    ICudaImageFilter() = delete;

    explicit ICudaImageFilter(CudaChannel &channel) : channel_(&channel) {
    }

    explicit ICudaImageFilter(const CudaChannel &cuda_channel) : channel_(const_cast<CudaChannel *>(&cuda_channel)) {
    }

    virtual ~ICudaImageFilter() = default;

    virtual void apply(const Image<Channels> &image) const = 0;

protected:
    CudaChannel *channel_;

    const NppStreamContext *getNppStreamContext() const;

    friend class CudaRuntime;
    friend class CudaChannel;
};
