#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "YOLOInference.h"
#include "FFmpegStream.h"  // Assuming the YOLOInference and FFmpegStream classes are in separate headers

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
    try {
        // Argument Parsing
        std::string model_path, input_url;
        float conf_thresh = 0.25, iou_thresh = 0.45;
        std::vector<int> imgsz = {640, 640};
        bool view_img = false;

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("weights", po::value<std::string>(&model_path)->default_value("yolov5.onnx"), "model path")
            ("source", po::value<std::string>(&input_url)->default_value("rtsp://your_stream_url"), "input video stream URL")
            ("imgsz", po::value<std::vector<int>>(&imgsz)->multitoken(), "inference size (height, width)")
            ("conf-thres", po::value<float>(&conf_thresh)->default_value(0.25), "confidence threshold")
            ("iou-thres", po::value<float>(&iou_thresh)->default_value(0.45), "IOU threshold")
            ("view-img", po::bool_switch(&view_img), "show results");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 1;
        }

        // Initialize YOLO Inference and Stream Handler
        YOLOInference yolo(model_path);
        FFmpegStream stream(input_url);

        while (true) {
            cv::Mat frame = stream.readFrame();
            std::vector<float> detections = yolo.infer(frame);

            // Draw bounding boxes (placeholder for actual post-processing)
            for (size_t i = 0; i < detections.size(); i += 6) {
                int x = static_cast<int>(detections[i] * frame.cols);
                int y = static_cast<int>(detections[i + 1] * frame.rows);
                int w = static_cast<int>(detections[i + 2] * frame.cols);
                int h = static_cast<int>(detections[i + 3] * frame.rows);

                cv::rectangle(frame, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
            }

            if (view_img) {
                cv::imshow("YOLOv5 Inference", frame);
                if (cv::waitKey(1) == 27) break; // Press 'ESC' to exit
            }
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    return 0;
}

