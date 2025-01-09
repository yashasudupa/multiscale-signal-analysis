#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp> // For image compression
#include "YOLOInference.h"
#include "FFmpegStream.h"  // Assuming the YOLOInference and FFmpegStream classes are in separate headers

namespace po = boost::program_options;

// Function for assembly optimization
void optimizedBoundingBox(float* detections, int num_detections, int frame_width, int frame_height) {
    // Example of inline assembly for optimization (x86-64 used for illustration; adjust for ARM or other architectures)
    asm volatile (
        "mov %[num], %%ecx\n\t"           // Load number of detections
        "1:\n\t"
        "movaps (%[det]), %%xmm0\n\t"    // Load detection data into SSE register
        "mulps %[width], %%xmm0\n\t"    // Scale coordinates with frame dimensions
        "sub $16, %[det]\n\t"           // Move to the next detection
        "loop 1b\n\t"                   // Loop until all detections processed
        :
        : [det] "r"(detections), [num] "r"(num_detections), [width] "m"(frame_width)
        : "ecx", "xmm0", "memory"
    );
}

int main(int argc, char* argv[]) {
    try {
        // Argument Parsing
        std::string model_path, input_url;
        float conf_thresh = 0.25f, iou_thresh = 0.45f; // Floating-point precision
        std::vector<int> imgsz = {640, 640};
        bool view_img = false;

        po::options_description desc("Allowed options");
        desc.add_options()
            ("help,h", "produce help message")
            ("weights", po::value<std::string>(&model_path)->default_value("yolov5.onnx"), "model path")
            ("source", po::value<std::string>(&input_url)->default_value("rtsp://your_stream_url"), "input video stream URL")
            ("imgsz", po::value<std::vector<int>>(&imgsz)->multitoken(), "inference size (height, width)")
            ("conf-thres", po::value<float>(&conf_thresh)->default_value(0.25f), "confidence threshold")
            ("iou-thres", po::value<float>(&iou_thresh)->default_value(0.45f), "IOU threshold")
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

            // **Image Compression** for efficient processing
            std::vector<uchar> compressed_frame;
            cv::imencode(".jpg", frame, compressed_frame); // Compress frame into JPEG
            cv::Mat decompressed_frame = cv::imdecode(compressed_frame, cv::IMREAD_COLOR); // Decompress

            // Perform Inference
            std::vector<float> detections = yolo.infer(decompressed_frame);

            // **Assembly Optimization** for bounding box calculations
            optimizedBoundingBox(detections.data(), detections.size() / 6, frame.cols, frame.rows);

            // Draw bounding boxes
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
