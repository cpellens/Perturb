#pragma once

#include "common.h"
#include "CudaChannel.h"

class GrayscaleFilter {
public:
    GrayscaleFilter() = delete;

    explicit GrayscaleFilter(CudaChannel &channel) : channel_(channel) {
    }

    template<int Channels>
    void apply(const NppiSize &size) const;

private:
    CudaChannel &channel_;
};

// 0xb1a800000
// 0xb1ce00000
// b1c2
