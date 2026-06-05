#pragma once
#include "Image.h"

class CudaChannel;

template<int Channels>
class ICudaImageFilter {
public:
    ICudaImageFilter() = delete;

    explicit ICudaImageFilter(CudaChannel &channel) : channel_(&channel) {
    };

    virtual ~ICudaImageFilter() = default;

    virtual void apply(const Image<Channels> &image) const = 0;

    CudaChannel *channel_;

    friend class CudaRuntime;
    friend class CudaChannel;
};
