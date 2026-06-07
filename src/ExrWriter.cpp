#define TINYEXR_USE_STB_ZLIB 1
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_IMPLEMENTATION 1
#define TINYEXR_USE_COMPILER_FP16 1
#include <tinyexr.h>

#include "ExrWriter.h"
#include <ranges>
#include <algorithm>

void ExrWriter::write(const Image<1> &image, const std::string &filename) { write<1>(image, filename); }

void ExrWriter::write(const Image<2> &image, const std::string &filename) { write<2>(image, filename); }

void ExrWriter::write(const Image<3> &image, const std::string &filename) { write<3>(image, filename); }

void ExrWriter::write(const Image<4> &image, const std::string &filename) { write<4>(image, filename); }

void ExrWriter::initializeExrHeader(const int channels) {
    exr_header_ = std::make_shared<EXRHeader>();
    exr_header_->num_channels = channels;
    exr_header_->pixel_aspect_ratio = 1.0f;
    exr_header_->compression_type = TINYEXR_COMPRESSIONTYPE_NONE;
}

void ExrWriter::initializeExrImage(const int channels) {
    exr_image_ = std::make_shared<EXRImage>();
    exr_image_->num_channels = channels;
    exr_image_->width = 0;
    exr_image_->height = 0;
}

template<uint8_t Channels>
void ExrWriter::write(const Image<Channels> &image, const std::string &filename) {
    initializeExrHeader(Channels);
    initializeExrImage(Channels);

    // Update the image metadata
    exr_image_->width = image.getWidth();
    exr_image_->height = image.getHeight();

    {
        // Separate the image channels into pixel buffers
        std::array<Npp32f *, Channels> pixelBuffers{};
        preparePixelBuffers<Channels>(image, pixelBuffers);

        // The images in the exr_image must be an unsigned char** instead of float**
        exr_image_->images = new Npp8u *[Channels];
        for (int c = 0; c < Channels; ++c) {
            exr_image_->images[c] = new Npp8u[image.getWidth() * image.getHeight() * sizeof(Npp32f)];
            std::memmove(exr_image_->images[c], pixelBuffers[c], image.getWidth() * image.getHeight() * sizeof(Npp32f));
        }
    }

    // Update the header with channel data
    exr_header_->channels = new EXRChannelInfo[Channels];
    exr_header_->pixel_types = new int[Channels];
    exr_header_->requested_pixel_types = new int[Channels];

    static std::string channelNames[] = {"R", "G", "B", "A"};
    for (int channel = 0; channel < Channels; ++channel) {
        auto const channelPtr = &exr_header_->channels[channel];

        auto const channelName = channelNames[channel] + "\0";
        strncpy_s(channelPtr->name, channelName.c_str(), channelName.length());

        // Request to store the channel as half-precision
        exr_header_->pixel_types[channel] = TINYEXR_PIXELTYPE_FLOAT;
        exr_header_->requested_pixel_types[channel] = TINYEXR_PIXELTYPE_HALF;
    }

    // Attempt to write the EXR file
    const char *errMsg = nullptr;
    if (auto const status = SaveEXRImageToFile(exr_image_.get(), exr_header_.get(), filename.c_str(), &errMsg);
        status != TINYEXR_SUCCESS) {
        throw std::runtime_error(errMsg);
    }

    // Clean up EXR headers and image resources
    delete[] exr_header_->channels;
    delete[] exr_header_->pixel_types;
    delete[] exr_header_->requested_pixel_types;
    delete[] exr_image_->images;

    // Free the EXR header and image
    exr_header_.reset();
    exr_image_.reset();
}

template<uint8_t Channels>
void ExrWriter::preparePixelBuffers(const Image<Channels> &image, std::array<Npp32f *, Channels> &pixelBuffers) {
    auto const pixels = image.getPixels();
    auto const pixelCount = image.getWidth() * image.getHeight();

    // Just a sanity check before we begin
    if (pixelCount * Channels != pixels.size()) {
        throw std::runtime_error(
            "Invalid pixel count; Expected " + std::to_string(pixelCount * Channels) + " but got " + std::to_string(
                pixels.size()));
    }

    std::vector<Npp32f> channel(pixelCount, 0.0f);

    for (int c = 0; c < Channels; ++c) {
        std::generate_n(channel.begin(), pixelCount,
                        [&pixels, c, i = 0]() mutable {
                            const int idx = i++ * Channels + c;
                            if (idx >= pixels.size()) {
                                throw std::runtime_error("Index out of bounds at " + std::to_string(idx));
                            }

                            return pixels[idx];
                        });

        pixelBuffers[c] = new Npp32f[pixelCount];
        std::ranges::copy(channel, pixelBuffers[c]);
    }
}

template void ExrWriter::write<1>(const Image<1> &image, const std::string &filename);

template void ExrWriter::write<2>(const Image<2> &image, const std::string &filename);

template void ExrWriter::write<3>(const Image<3> &image, const std::string &filename);

template void ExrWriter::write<4>(const Image<4> &image, const std::string &filename);
