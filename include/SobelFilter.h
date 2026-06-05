#pragma once

#include "ICudaImageFilter.h"
#include "Image.h"

template<int Channels>
class SobelFilter final : public ICudaImageFilter<Channels> {
public:
    enum Direction {
        Horizontal,
        Vertical,
    };

    explicit SobelFilter(CudaChannel &channel) : ICudaImageFilter<Channels>(channel) {
    }

    void apply(const Image<Channels> &image, Direction direction) const;

    void apply(const Image<Channels> &image) const override;
};
