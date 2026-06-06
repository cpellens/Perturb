#include <iostream>

#include "CudaChannel.h"
#include "CudaRuntime.h"
#include "Image.h"
#include "NormalMapFilter.h"
#include "Timer.h"

int main(const int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image> [<...>]\n";
        return 1;
    }

#ifdef _DEBUG
    auto const timer = std::make_unique<Timer>(std::cerr);
#else
    auto const timer = std::make_unique<Timer>();
#endif

    timer->start("init cuda runtime");
    const auto runtime = std::make_unique<CudaRuntime>();
    std::cout << "Using CUDA device: " << runtime->getDeviceName() << std::endl;
    const auto channel = std::make_unique<CudaChannel>(*runtime);
    constexpr int SourceChannels = 3;

    // Create the filter pipeline
    timer->lap("create filter pipeline");
    const NormalMapFilter<SourceChannels> normalMap(*channel);

    for (int i = 1; i < argc; ++i) {
        const std::filesystem::path imagePath(argv[i]);
        if (!imagePath.has_filename() || !exists(imagePath)) {
            std::cerr << "Invalid image path: " << imagePath << std::endl;
            return 1;
        }

        // Get the absolute path and the base filename
        auto const imagePathAbsolute = absolute(imagePath);
        auto const baseFileName = imagePathAbsolute.stem();

        std::cout << "Using image: " << imagePathAbsolute << std::endl;
        auto const outputPathAbsolute = imagePathAbsolute.parent_path() / (baseFileName.string() + "-normal-map.png");

        // Load the image
        timer->lap("load image");
        Image<SourceChannels> image(imagePathAbsolute.string());

        // Write the image to the channel
        timer->lap("image -> channel");
        image.writeToChannel(*channel);
        runtime->synchronize();

        try {
            // Create the normal map
            timer->lap("create normal map");
            normalMap.apply(image);

            // One last image for the normal map
            timer->lap("normal map -> host");
            image.readFromChannel(*channel);

            runtime->synchronize();

            image.save<uint8_t>(outputPathAbsolute.string());
            std::cout << "Saved normal map to: " << outputPathAbsolute << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    timer->stop();
    channel->freeDevicePtr();
    cudaDeviceReset();

    return 0;
}
