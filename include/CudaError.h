#pragma once

#include <map>

#include "common.h"

/* ----------------------------------------------------------------------
 *  Map from NPP status codes to human‑readable messages.
 *  All values from the list you posted are represented.
 * ---------------------------------------------------------------------- */
inline std::map<NppStatus, std::string> errorMessages{
    /* ---- Errors (negative values) ----------------------------------- */
    {NPP_NOT_SUPPORTED_MODE_ERROR, "Mode not supported"},
    {NPP_LIBRARY_VERSION_MISMATCH_ERROR, "Library version mismatch"},
    {NPP_INVALID_HOST_POINTER_ERROR, "Invalid host pointer"},
    {NPP_INVALID_DEVICE_POINTER_ERROR, "Invalid device pointer"},
    {NPP_LUT_PALETTE_BITSIZE_ERROR, "LUT palette bit size error"},
    {NPP_ZC_MODE_NOT_SUPPORTED_ERROR, "Zero‑crossing mode not supported"},
    {NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY, "Insufficient compute capability"},
    {NPP_TEXTURE_BIND_ERROR, "Texture bind error"},
    {NPP_WRONG_INTERSECTION_ROI_ERROR, "Wrong intersection ROI"},
    {NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR, "Haar classifier pixel match error"},
    {NPP_MEMFREE_ERROR, "Memory free error"},
    {NPP_MEMSET_ERROR, "Memory set error"},
    {NPP_MEMCPY_ERROR, "Memory copy error"},
    {NPP_ALIGNMENT_ERROR, "Alignment error"},
    {NPP_CUDA_KERNEL_EXECUTION_ERROR, "CUDA kernel execution error"},
    {NPP_STREAM_CTX_ERROR, "Stream context error"},
    {NPP_ROUND_MODE_NOT_SUPPORTED_ERROR, "Round mode not supported"},
    {NPP_QUALITY_INDEX_ERROR, "Quality index error"},
    {NPP_RESIZE_NO_OPERATION_ERROR, "Resize no operation"},
    {NPP_OVERFLOW_ERROR, "Overflow error"},
    {NPP_NOT_EVEN_STEP_ERROR, "Not even step error"},
    {NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR, "Histogram number of levels error"},
    {NPP_LUT_NUMBER_OF_LEVELS_ERROR, "LUT number of levels error"},
    {NPP_CORRUPTED_DATA_ERROR, "Corrupted data error"},
    {NPP_CHANNEL_ORDER_ERROR, "Channel order error"},
    {NPP_ZERO_MASK_VALUE_ERROR, "Zero mask value error"},
    {NPP_QUADRANGLE_ERROR, "Quadrangle error"},
    {NPP_RECTANGLE_ERROR, "Rectangle error"},
    {NPP_COEFFICIENT_ERROR, "Coefficient error"},
    {NPP_NUMBER_OF_CHANNELS_ERROR, "Number of channels error"},
    {NPP_COI_ERROR, "Channel of interest error"},
    {NPP_DIVISOR_ERROR, "Divisor error"},
    {NPP_CHANNEL_ERROR, "Channel error"},
    {NPP_STRIDE_ERROR, "Stride error"},
    {NPP_ANCHOR_ERROR, "Anchor error"},
    {NPP_MASK_SIZE_ERROR, "Mask size error"},
    {NPP_RESIZE_FACTOR_ERROR, "Resize factor error"},
    {NPP_INTERPOLATION_ERROR, "Interpolation error"},
    {NPP_MIRROR_FLIP_ERROR, "Mirror/flip error"},
    {NPP_MOMENT_00_ZERO_ERROR, "Moment 0 zero error"},
    {NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR, "Threshold negative level error"},
    {NPP_THRESHOLD_ERROR, "Threshold error"},
    {NPP_CONTEXT_MATCH_ERROR, "Context match error"},
    {NPP_FFT_FLAG_ERROR, "FFT flag error"},
    {NPP_FFT_ORDER_ERROR, "FFT order error"},
    {NPP_STEP_ERROR, "Step error"},
    {NPP_SCALE_RANGE_ERROR, "Scale range error"},
    {NPP_DATA_TYPE_ERROR, "Data type error"},
    {NPP_OUT_OFF_RANGE_ERROR, "Out of range error"},
    {NPP_DIVIDE_BY_ZERO_ERROR, "Divide by zero error"},
    {NPP_MEMORY_ALLOCATION_ERR, "Memory allocation error"},
    {NPP_NULL_POINTER_ERROR, "Null pointer error"},
    {NPP_RANGE_ERROR, "Range error"},
    {NPP_SIZE_ERROR, "Size error"},
    {NPP_BAD_ARGUMENT_ERROR, "Bad argument error"},
    {NPP_NO_MEMORY_ERROR, "No memory error"},
    {NPP_NOT_IMPLEMENTED_ERROR, "Not implemented error"},
    {NPP_ERROR, "General error"},
    {NPP_ERROR_RESERVED, "Reserved error"},

    /* ---- No‑error / success ---------------------------------------- */
    {NPP_NO_ERROR, "No error"},
    {NPP_SUCCESS, "Success"}, // same as NPP_NO_ERROR

    /* ---- Warnings (positive values) ------------------------------- */
    {NPP_NO_OPERATION_WARNING, "No operation performed"},
    {NPP_DIVIDE_BY_ZERO_WARNING, "Divide by zero warning"},
    {NPP_AFFINE_QUAD_INCORRECT_WARNING, "Affine quadrilateral incorrect"},
    {NPP_WRONG_INTERSECTION_ROI_WARNING, "Wrong intersection ROI warning"},
    {NPP_WRONG_INTERSECTION_QUAD_WARNING, "Wrong intersection quadrilateral warning"},
    {NPP_DOUBLE_SIZE_WARNING, "Double size warning"},
    {NPP_MISALIGNED_DST_ROI_WARNING, "Misaligned destination ROI warning"}
};

inline std::string &nppGetErrorString(const NppStatus &status) {
    static std::string defaultError = "Unknown error";
    if (!errorMessages.contains(status)) {
        return defaultError;
    }
    return errorMessages.at(status);
}

inline void handleNppError(const NppStatus &status) {
    static std::stringstream errMsg;

    if (status != NPP_SUCCESS) {
        errMsg.str("");

        errMsg << "NPP Error: " << nppGetErrorString(status) << '\n';
        throw std::runtime_error(errMsg.str());
    }
}

inline void handleCudaError(const cudaError_t &cudaErrorCode) {
    if (cudaErrorCode != cudaSuccess) {
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(cudaErrorCode)));
    }
}
