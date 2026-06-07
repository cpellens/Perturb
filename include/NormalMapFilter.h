#pragma once

#include "CudaChannel.h"
#include "GaussianBlurFilter.h"
#include "GrayscaleFilter.h"
#include "ICudaImageFilter.h"
#include "Image.h"
#include "SobelFilter.h"

template<int Channels>
class NormalMapFilter final : public ICudaImageFilter<Channels> {
public:
    NormalMapFilter() = delete;

    explicit NormalMapFilter(CudaChannel &channel, NppiMaskSize blurKernelSize = NPP_MASK_SIZE_3_X_3);

    void apply(const Image<1> &a, const Image<1> &b) const;

    void apply(const Image<Channels> &image) const override;

private:
    SobelFilter<1> sobelFilter;
    GaussianBlurFilter<1> gaussianBlurFilter;
    GrayscaleFilter<Channels> grayscaleFilter;
};
