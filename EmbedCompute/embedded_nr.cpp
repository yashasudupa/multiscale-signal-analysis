#include "EmbeddedNR.h"

/**
 * @brief Applies median filtering to reduce noise in the image.
 * @param src The source image to be denoised.
 * @param dst The destination image where the denoised result is stored.
 * @param kernelSize The size of the median filter kernel. Must be odd; if even, it's increased by 1.
 */
#include "EmbeddedNR.h"

namespace EmbeddedNR {
    void applyMedianFilter(cv::Mat& src, cv::Mat& dst, int kernelSize = 3) {
        // Ensure the kernel size is odd
        if (kernelSize % 2 == 0) {
            kernelSize++;
        }
        cv::medianBlur(src, dst, kernelSize);
    }
}