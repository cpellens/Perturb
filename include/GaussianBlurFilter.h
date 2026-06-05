#pragma once

#include "Image.h"

class GaussianBlurFilter {
public:
    GaussianBlurFilter() = delete;

    explicit GaussianBlurFilter(CudaChannel &channel,
                                NppiMaskSize kernelSize) noexcept;

    template<int Channels>
    void apply(const Image<Channels> &image) const;

private:
    NppiMaskSize kernelSize_;
    CudaChannel &channel_;
};
