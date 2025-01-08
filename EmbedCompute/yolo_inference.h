#ifndef YOLOINFERENCE_H
#define YOLOINFERENCE_H

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class YOLOInference {
public:
    YOLOInference(const std::string& model_path);
    std::vector<float> infer(const cv::Mat& frame);

private:
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<int64_t> input_shape_;
};

#endif
