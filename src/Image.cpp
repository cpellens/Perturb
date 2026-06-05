#include "Image.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>
#else
#include <stb_image.h>
#endif

template<uint8_t CHANNELS>
Image<CHANNELS>::Image(const int w, const int h) noexcept : width(w), height(h), image(w * h * CHANNELS, 0.0f) {
}

template<uint8_t CHANNELS>
Image<CHANNELS>::Image(const std::string &filename) : Image(filename.c_str()) {
}

template<uint8_t CHANNELS>
Image<CHANNELS>::Image(const char *filename) {
    int n;
    auto const im = stbi_loadf(filename, &width, &height, &n, CHANNELS);
    if (!im) {
        std::cerr << "Failed to load image " << filename << std::endl;
        return;
    }
    if (n != CHANNELS) {
        stbi_image_free(im);
        std::stringstream errMsg;
        errMsg << "Image " << filename << " has " << n << " channels, expected " << static_cast<int>(CHANNELS);
        throw std::invalid_argument(errMsg.str());
    }

    // Convert to float and store in our image vector
    image.resize(width * height * CHANNELS);
    std::ranges::copy_n(im, width * height * CHANNELS, image.begin());

    stbi_image_free(im);
}

template<uint8_t CHANNELS>
Image<CHANNELS>::Image(const Image &other) noexcept : width(other.width), height(other.height), image(other.image) {
    image.resize(width * height * CHANNELS);

    // Copy the pixels from the other image to our image vector
    std::ranges::copy_n(other.image.begin(), width * height * CHANNELS, image.begin());
}

template<uint8_t CHANNELS>
void Image<CHANNELS>::save(const char *filename) const {
    if (!filename) {
        throw std::invalid_argument("Filename cannot be null");
    }

    if (stbi_write_hdr(filename, width, height, CHANNELS, image.data()) == 0) {
        throw std::runtime_error("Failed to save HDR image");
    }
}

template<uint8_t CHANNELS>
template<typename T>
void Image<CHANNELS>::save(const char *filename) const {
    if (!filename) {
        throw std::invalid_argument("Filename cannot be null");
    }

    // We want to convert floats to the targeted type
    std::vector<T> convertedImage(width * height * CHANNELS);

    auto constexpr maxValueOfT = std::numeric_limits<T>::max();
    std::ranges::transform(image.begin(), image.end(), convertedImage.begin(), [&](const float f) {
        auto const clamped = std::clamp(f, 0.0f, 1.0f);
        return static_cast<T>(clamped * maxValueOfT);
    });

    if (stbi_write_png(filename, width, height, CHANNELS, convertedImage.data(), width * CHANNELS * sizeof(T)) == 0) {
        throw std::runtime_error("Failed to save PNG image");
    }
}

template<uint8_t CHANNELS>
template<typename T>
void Image<CHANNELS>::save(const std::string &filename) const {
    save<T>(filename.c_str());
}

template<uint8_t CHANNELS>
void Image<CHANNELS>::save(const std::string &filename) const {
    save(filename.c_str());
}

template<uint8_t CHANNELS>
int Image<CHANNELS>::getWidth() const noexcept {
    return width;
}

template<uint8_t CHANNELS>
int Image<CHANNELS>::getHeight() const noexcept {
    return height;
}

template<uint8_t CHANNELS>
std::span<Npp32f> Image<CHANNELS>::getPixelsMutable() noexcept {
    return image;
}

template<uint8_t CHANNELS>
NppiSize Image<CHANNELS>::getSize() const noexcept {
    return {width, height};
}

template<uint8_t CHANNELS>
void Image<CHANNELS>::writeToChannel(CudaChannel &channel) const {
    channel.setDevicePtr(createDevicePtr(channel), width * sizeof(Npp32f) * CHANNELS * height);
}

template<uint8_t CHANNELS>
void Image<CHANNELS>::readFromChannel(const CudaChannel &channel) {
    auto constexpr bytesPerPixel = sizeof(Npp32f) * CHANNELS;
    auto const nppStreamCtx = &channel.getNppStreamContext();
    if (!nppStreamCtx) {
        throw std::invalid_argument("NPP stream context is null");
    }

    auto const cudaStream = nppStreamCtx->hStream;
    if (!cudaStream) {
        throw std::invalid_argument("CUDA stream is null");
    }

    // Get the existing device ptr
    auto const devicePtr = channel.getDevicePtr();
    if (!devicePtr) {
        throw std::invalid_argument("Device pointer is null");
    }

    const auto nSrcStep = width * bytesPerPixel;
    image.resize(width * height * CHANNELS);

    // Copy the memory from the devicePtr to the image.data() host pointer
    if (cudaMemcpy2DAsync(
            image.data(),
            nSrcStep,
            devicePtr,
            nSrcStep,
            nSrcStep,
            height,
            cudaMemcpyDeviceToHost,
            cudaStream) != cudaSuccess) {
        auto const cudaErrorCode = cudaGetLastError();
        auto const errMsg = std::string(cudaGetErrorString(cudaErrorCode));
        throw std::runtime_error("Failed to copy image data from device: " + errMsg);
    }
}

template<uint8_t CHANNELS>
Npp32f *Image<CHANNELS>::createDevicePtr(const CudaChannel &channel) const {
    // Get NPP stream context
    auto const nppStreamCtx = &channel.getNppStreamContext();
    if (!nppStreamCtx) {
        throw std::invalid_argument("NPP stream context is null");
    }

    // Get CUDA stream context
    auto const cudaStream = nppStreamCtx->hStream;
    if (!cudaStream) {
        throw std::invalid_argument("CUDA stream is null");
    }

    // Allocate device memory for the image data
    void *devicePtr = nullptr;
    size_t devicePitch;
    if (cudaMallocPitch(&devicePtr, &devicePitch, width * sizeof(Npp32f) * CHANNELS, height) != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory");
    }

    // Copy the memory from the image.data() host pointer to the devicePtr
    if (cudaMemcpy2DAsync(
            devicePtr,
            devicePitch,
            image.data(),
            devicePitch,
            width * sizeof(Npp32f) * CHANNELS,
            height,
            cudaMemcpyHostToDevice, cudaStream) != cudaSuccess) {
        auto const cudaErrorCode = cudaGetLastError();
        auto const errMsg = std::string(cudaGetErrorString(cudaErrorCode));
        throw std::runtime_error("Failed to copy image data to device: " + errMsg);
    }

    return static_cast<Npp32f *>(devicePtr);
}

template class Image<1>;
template class Image<2>;
template class Image<3>;
template class Image<4>;

template void Image<1>::save<uint8_t>(const std::string &filename) const;

template void Image<2>::save<uint8_t>(const std::string &filename) const;

template void Image<3>::save<uint8_t>(const std::string &filename) const;

template void Image<4>::save<uint8_t>(const std::string &filename) const;
