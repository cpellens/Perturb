#pragma once

#include "CudaChannel.h"
#include "Image.h"

class NormalMapFilter {
public:
    NormalMapFilter() = delete;

    explicit NormalMapFilter(CudaChannel &channel)
        : channel_(channel) {
    }

    void apply(const Image<1> &a, const Image<1> &b) const;

private:
    CudaChannel &channel_;
};
