#pragma once

#include "common.h"
#include "CudaChannel.h"
#include "ICudaImageFilter.h"

template<int Channels>
class GrayscaleFilter final : public ICudaImageFilter<Channels> {
public:
    explicit GrayscaleFilter(CudaChannel &channel) : ICudaImageFilter<Channels>(channel) {
    }

    ~GrayscaleFilter() override = default;

    void apply(const Image<Channels> &image) const override;
};
