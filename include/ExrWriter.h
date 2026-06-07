#pragma once
#include <string>

#include "Image.h"
#include <tinyexr.h>

#include "IImageWriter.h"

class ExrWriter final : public IImageWriter {
public:
    ExrWriter() = default;

    template<uint8_t Channels>
    void write(const Image<Channels> &image, const std::string &filename);

    void write(const Image<1> &image, const std::string &filename) override;

    void write(const Image<2> &image, const std::string &filename) override;

    void write(const Image<3> &image, const std::string &filename) override;

    void write(const Image<4> &image, const std::string &filename) override;

private:
    std::shared_ptr<EXRHeader> exr_header_;
    std::shared_ptr<EXRImage> exr_image_;

    void initializeExrHeader(int channels);

    void initializeExrImage(int channels);

    template<uint8_t Channels>
    static void preparePixelBuffers(const Image<Channels> &image, std::array<Npp32f *, Channels> &pixelBuffers);
};
