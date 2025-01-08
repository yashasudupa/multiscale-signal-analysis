#include "yolo_inference.h"

YOLOInference::YOLOInference(const std::string& model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv5");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
    input_shape_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<float> YOLOInference::infer(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_shape_[2], input_shape_[3]));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<float> input_tensor(resized.begin<float>(), resized.end<float>());

    std::vector<const char*> input_names = { session_->GetInputName(0, allocator_) };
    std::vector<const char*> output_names = { session_->GetOutputName(0, allocator_) };
    std::vector<Ort::Value> input_tensors;

    input_tensors.emplace_back(Ort::Value::CreateTensor<float>(allocator_, input_tensor.data(),
                                                               input_tensor.size(), input_shape_.data(), input_shape_.size()));

    auto output_tensors = session_->Run(Ort::RunOptions{ nullptr }, input_names.data(), input_tensors.data(), 1,
                                        output_names.data(), 1);

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount());
}
