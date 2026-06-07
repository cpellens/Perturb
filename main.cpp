#include <iostream>

#include "CudaChannel.h"
#include "CudaRuntime.h"
#include "ExrWriter.h"
#include "Image.h"
#include "NormalMapFilter.h"
#include "Timer.h"

#include <omp.h>

int main(const int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image> [<...>]\n";
        return 1;
    }

    constexpr int SourceChannels = 3;

#ifdef _DEBUG
    auto const timer = std::make_unique<Timer>(std::cerr);
#else
    auto const timer = std::make_unique<Timer>();
#endif

    timer->start("init cuda runtime");
    const auto runtime = std::make_unique<CudaRuntime>();
    std::cout << "Using CUDA device: " << runtime->getDeviceName() << std::endl;

    const auto channel = std::make_unique<CudaChannel>(*runtime);
    const NormalMapFilter<SourceChannels> normalMap(*channel);
    std::vector<std::tuple<Image<SourceChannels>, std::string> > images{};
    images.reserve(argc);

    timer->beginLap("load images");

#pragma omp parallel for shared(argv, argc, images)
    for (int i = 1; i < argc; ++i) {
        const std::filesystem::path imagePath(argv[i]);
        if (!imagePath.has_filename() || !exists(imagePath)) {
            std::cerr << "Invalid image path: " << imagePath << std::endl;
            continue;
        }

        // Get the absolute path and the base filename
        auto const imagePathAbsolute = absolute(imagePath);
        auto const baseFileName = imagePathAbsolute.stem();
        auto const outputPath = imagePathAbsolute.parent_path() / (baseFileName.string() + "-normal-map.exr");

        // Load the image
        Image<SourceChannels> image(imagePathAbsolute.string());
        images.emplace_back(std::move(image), outputPath.string());

        std::cout << "Loaded image: " << imagePathAbsolute << std::endl;
    }

    try {
        // Write the image to the channel
        // (Host memory -> Device memory)

        for (auto &[image, outputPath]: images) {
            timer->beginLap("image -> channel");
            image.writeToChannel(*channel);
            runtime->synchronize();

            // Create the normal map
            // Executes the normal map filtering on the device
            // then writes back to device memory
            timer->beginLap("create normal map");
            normalMap.apply(image);

            // Read the image back from the device memory into host memory.
            // The normal map will always have 3 dimensions. Nobody knows why. It should be 2.
            timer->beginLap("normal map -> host");
            image.readFromChannel(*channel);

            // Wait for the device to complete
            runtime->synchronize();
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    timer->beginLap("write images");
    ExrWriter exrWriter;

#pragma omp parallel for shared(images) private(exrWriter)
    for (int i = 0; i < images.size(); ++i) {
        auto const [image, outputPath] = images[i];
        exrWriter.write(image, outputPath);
        std::cout << "Wrote image: " << outputPath << std::endl;
    }

    timer->stop();

    return 0;
}
