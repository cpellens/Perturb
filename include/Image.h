#pragma once

#include "common.h"
#include "CudaChannel.h"

template<uint8_t Channels>
class Image {
    int width{}, height{};
    std::vector<Npp32f> image;

    const uint8_t channels = Channels;

public:
    Image() = delete;

    Image(int w, int h) noexcept;

    explicit Image(const char *filename);

    explicit Image(const std::string &filename);

    explicit Image(const Image &other) noexcept;

    void save(const char *filename) const;

    void save(const std::string &filename) const;

    template<typename T>
    void save(const char *filename) const;

    template<typename T>
    void save(const std::string &filename) const;

    [[nodiscard]] int getWidth() const noexcept;

    [[nodiscard]] int getHeight() const noexcept;

    [[nodiscard]] static constexpr uint8_t getChannels() noexcept;

    [[nodiscard]] std::span<const Npp32f> getPixels() const noexcept;

    std::span<Npp32f> getPixelsMutable() noexcept;

    [[nodiscard]] NppiSize getSize() const noexcept;

    void writeToChannel(CudaChannel &channel) const;

    void readFromChannel(const CudaChannel &channel);

    ~Image() = default;

    [[nodiscard]] Npp32f *createDevicePtr(const CudaChannel &channel) const;
};

template<uint8_t Channels>
constexpr uint8_t Image<Channels>::getChannels() noexcept {
    return Channels;
}
