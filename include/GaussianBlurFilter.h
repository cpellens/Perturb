#pragma once

#include "ICudaImageFilter.h"
#include "Image.h"

template<int Channels>
class GaussianBlurFilter final : public ICudaImageFilter<Channels> {
public:
    GaussianBlurFilter() = delete;

    explicit GaussianBlurFilter(CudaChannel &channel, NppiMaskSize kernelSize) noexcept;

    void apply(const Image<Channels> &image) const override;

protected:
    NppiMaskSize kernelSize_;
};
