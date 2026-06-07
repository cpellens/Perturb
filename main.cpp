#include <iostream>

#include "CudaChannel.h"
#include "CudaRuntime.h"
#include "ExrWriter.h"
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
    const auto exrWriter = std::make_unique<ExrWriter>();
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
        auto const outputPathAbsolute = imagePathAbsolute.parent_path() / (baseFileName.string() + "-normal-map.exr");

        std::cout << "Using image: " << imagePathAbsolute << std::endl;

        // Load the image
        timer->lap("load image");
        Image<SourceChannels> image(imagePathAbsolute.string());

        try {
            // Write the image to the channel
            // (Host memory -> Device memory)
            timer->lap("image -> channel");
            image.writeToChannel(*channel);
            runtime->synchronize();

            // Create the normal map
            // Executes the normal map filtering on the device
            // then writes back to device memory
            timer->lap("create normal map");
            normalMap.apply(image);

            // Read the image back from the device memory into host memory.
            // The normal map will always have 3 dimensions. Nobody knows why. It should be 2.
            timer->lap("normal map -> host");
            image.readFromChannel(*channel);

            // Wait for the device to complete
            runtime->synchronize();

            // image.save<uint8_t>(outputPathAbsolute.string());
            exrWriter->write < 3 > (image, outputPathAbsolute.string());
            std::cout << "Saved normal map to: " << outputPathAbsolute << std::endl;
        } catch (const std::exception &e) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
    }

    timer->stop();

    return 0;
}
