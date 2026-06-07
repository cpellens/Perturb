#pragma once
#include <string>

#include "Image.h"

class IImageWriter {
public:
    virtual ~IImageWriter() = default;

    virtual void write(const Image<1> &image, const std::string &filename) = 0;

    virtual void write(const Image<2> &image, const std::string &filename) = 0;

    virtual void write(const Image<3> &image, const std::string &filename) = 0;

    virtual void write(const Image<4> &image, const std::string &filename) = 0;
};
