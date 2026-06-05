#pragma once

#include "Image.h"

class SobelFilter {
public:
    enum Direction {
        Horizontal,
        Vertical,
    };

    explicit SobelFilter(CudaChannel &channel);

    template<int Channels>
    void apply(const Image<Channels> &image, Direction direction) const;

private:
    CudaChannel &channel_;
};
