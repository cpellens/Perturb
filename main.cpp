#include <iostream>
#include <vector>

#include "CudaChannel.h"
#include "CudaRuntime.h"
#include "GaussianBlurFilter.h"
#include "GrayscaleFilter.h"
#include "Image.h"
#include "NormalMapFilter.h"
#include "SobelFilter.h"
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
    const CudaRuntime runtime;
    std::cout << "Using CUDA device: " << runtime.getDeviceName() << std::endl;
    CudaChannel channel(runtime);

    for (int i = 1; i < argc; ++i) {
        constexpr int SourceChannels = 3;
        const std::filesystem::path imagePath(argv[i]);

        if (!imagePath.has_filename() || !exists(imagePath)) {
            std::cerr << "Invalid image path: " << imagePath << std::endl;
            return 1;
        }

        auto const imagePathAbsolute = absolute(imagePath);
        auto const imagePathAbsoluteStr = imagePathAbsolute.string();
        auto const baseFileName = imagePathAbsolute.stem();

        std::cout << "Using image: " << imagePathAbsolute << std::endl;
        auto const outputPathAbsolute = std::filesystem::absolute(
            imagePathAbsolute.parent_path().append(baseFileName.generic_string() + "-normal-map.png"));

        const Image<SourceChannels> sobelX(imagePathAbsoluteStr);
        const Image sobelY(sobelX); // NOLINT(*-unnecessary-copy-initialization)

        // Allocate space for the grayscale images
        Image<1> grayscaleX(sobelX.getWidth(), sobelX.getHeight());
        Image grayscaleY(grayscaleX);

        // Fetch the size of the input image
        auto const imageSize = sobelX.getSize();

        // Create the filter pipeline
        timer->lap("create filter pipeline");
        const SobelFilter filter(channel);
        const GaussianBlurFilter blurFilter(channel, NPP_MASK_SIZE_3_X_3);
        const GrayscaleFilter grayscale(channel);
        const NormalMapFilter normalMap(channel);

        // Apply the filter to the horizontal image
        timer->lap("horizontal -> device");
        sobelX.writeToChannel(channel);
        runtime.synchronize();

        timer->lap("apply horizontal sobel");
        filter.apply<SourceChannels>(sobelX, SobelFilter::Horizontal);

        timer->lap("soften result");
        blurFilter.apply<SourceChannels>(sobelX);

        timer->lap("grayscale horizontal result");
        grayscale.apply<SourceChannels>(imageSize);

        timer->lap("horizontal result -> host");
        grayscaleX.readFromChannel(channel);

        // Sync to ensure the GPU has finished processing the horizontal image
        // because the next application will create a new pointer
        runtime.synchronize();

        // Apply the filter to the vertical image
        sobelY.writeToChannel(channel);
        runtime.synchronize();

        timer->lap("apply vertical sobel");
        filter.apply<SourceChannels>(sobelY, SobelFilter::Vertical);

        timer->lap("soften result");
        blurFilter.apply<SourceChannels>(sobelY);

        timer->lap("grayscale vertical result");
        grayscale.apply<SourceChannels>(imageSize);

        timer->lap("vertical result -> host");
        grayscaleY.readFromChannel(channel);

        // Wait for the GPU to finish processing
        runtime.synchronize();

        // Create the normal map
        timer->lap("create normal map");
        normalMap.apply(grayscaleX, grayscaleY);

        // One last image for the normal map
        timer->lap("normal map -> host");

        Image<SourceChannels> normalMapImage(grayscaleX.getWidth(), grayscaleX.getHeight());
        normalMapImage.readFromChannel(channel);
        runtime.synchronize();

        normalMapImage.save<uint8_t>(outputPathAbsolute.string());
        std::cout << "Saved normal map to: " << outputPathAbsolute << std::endl;
    }

    return 0;
}
