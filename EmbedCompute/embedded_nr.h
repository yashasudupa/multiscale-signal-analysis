#ifndef EMBEDDEDNR_H
#define EMBEDDEDNR_H

#include <opencv2/opencv.hpp>

namespace EmbeddedNR {
    void applyMedianFilter(cv::Mat& src, cv::Mat& dst, int kernelSize = 3);
}

#endif // EMBEDDEDNR_H