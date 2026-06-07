#include <nppdefs.h>

__global__
void buildNormalMap(const Npp32f *gx,
                    const Npp32f *gy,
                    float3 *out,
                    size_t pitchBytes, // bytes per row
                    int width,
                    int height,
                    int channels);

/**
 * Launches a CUDA kernel to construct a normal map based on the gradient inputs.
 *
 * @param grid The grid dimensions for CUDA kernel execution.
 * @param block The block dimensions for CUDA kernel execution.
 * @param gx Pointer to the x-component gradient input.
 * @param gy Pointer to the y-component gradient input.
 * @param out Pointer to the output buffer for the calculated normal map.
 * @param pitch The pitch (row size in bytes) of the output buffer.
 * @param w The width of the input gradient data and output normal map. (in Pixels)
 * @param h The height of the input gradient data and output normal map. (in Pixels)
 * @param c The number of channels in the input images
 */
void launchBuildNormalMap(dim3 grid, dim3 block,
                          const Npp32f *gx, const Npp32f *gy,
                          float3 *out, size_t pitch, int w, int h, int c);


// Simple kernel that assumes:
// gx = -∂I/∂x, gy = -∂I/∂y,
//  nz = 1 (or whatever scale you like).

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define CLAMP(x, min, max) MIN(MAX(x, min), max)
#define CLAMP3(v, min, max) make_float3(CLAMP(v.x, min, max), CLAMP(v.y, min, max), CLAMP(v.z, min, max))

#ifdef __NVCC__
#include <cuda_runtime.h>

constexpr float3 upVector = float3{0.0f, 0.0f, 1.0f};

__global__
void buildNormalMap(const Npp32f *gx,
                    const Npp32f *gy,
                    float3 *out,
                    const size_t pitchBytes, // bytes per row
                    const int width,
                    const int height,
                    const int channels) {
    // 2‑D thread indices:
    // - Grid of blocks -> each block contains a grid of threads
    // - Block <-> Block: Global Memory
    // - Thread <-> Thread: Shared Memory

    const int blockOffsetX = blockIdx.x * blockDim.x;
    const int blockOffsetY = blockIdx.y * blockDim.y;

    const int x = blockOffsetX + threadIdx.x;
    const int y = blockOffsetY + threadIdx.y;

    if (x >= width || y >= height) return;

    const float divisor = 1.0f / channels;


    /* ---------- Access the 3‑channel float image --------------------
       The Sobel images are stored as 3 floats per pixel.
       For a normal map we usually only need one channel, e.g., luminance
       or the first component.
     */

    // Each row has 1024 pixels, with 3 channels each, as a float.
    // To get the source RGB, we must take the Y index and multiply it by the stride.
    // The stride is going to be channels * sizeof(float) * width.
    // The X offset will then be x * channels.
    const size_t rowPitchFloats = pitchBytes / sizeof(Npp32f); // often = width*3
    const size_t rowPitchRGB = pitchBytes / (sizeof(Npp32f) * channels);
    const int idxDst = y * rowPitchRGB + x;
    const int idxSrcStart = y * rowPitchFloats + x * channels;

    // Sum up the x and y values
    float xSum = 0.0f, ySum = 0.0f;
    if (channels > 1) {
        for (int i = 0; i < channels; i++) {
            xSum += gx[idxSrcStart + i];
            ySum += gy[idxSrcStart + i];
        }
    } else {
        xSum = gx[idxSrcStart];
        ySum = gy[idxSrcStart];
    }

    // horizontal gradient (negated → right‑hand rule)
    const float gxVal = -divisor * xSum;
    const float gyVal = -divisor * ySum;

    /* ------------------ Build the normal vector -------------------- */
    const float nz = 1.0f; // “height” component
    const float len2 = gxVal * gxVal + gyVal * gyVal + nz * nz;
    if (len2 <= 0.0001f) {
        // avoid division by zero
        out[idxDst] = upVector;
        return;
    }

    // normalize the normal vector using the inverse length
    const float invLen = rsqrtf(len2);

    float nx = gxVal * invLen;
    float ny = gyVal * invLen;
    float nz_norm = nz * invLen;

    // normalize the normal vector to the range [0, 1],
    // where 0.5 is the center of the texture
    nx = 0.5f + 0.5f * nx;
    ny = 0.5f + 0.5f * ny;

    out[idxDst] = CLAMP3(make_float3(nx, ny, nz_norm), 0.0f, 1.0f);
}

void launchBuildNormalMap(dim3 grid, dim3 block,
                          const Npp32f *gx, const Npp32f *gy,
                          float3 *out, size_t pitch, int w, int h, int c) {
    buildNormalMap<<<grid, block>>>(gx, gy, out, pitch, w, h, c);
}

#endif
