# Multi scale signal analysis

![image](https://github.com/user-attachments/assets/2d81388b-70e7-4195-b6e8-3e21e4bc02c6)

## Overview

This project involves interfacing an external camera with the Raspberry Pi 5 using FFmpeg, processing image data, and preparing datasets for object detection. It includes implementing data compression and optimizing night vision wildlife image detection using Assembly and C++. The project also explores future enhancements, such as integrating frequency domain analysis and real-time task scheduling with FreeRTOS.

## Project Structure
```css
   EmbedCompute/
   ├── build.sh                ← Build + setup + run automation
   ├── CMakeLists.txt          ← CMake configuration
   ├── requirements.txt        ← System-level packages
   ├── main.cpp                ← Entry point
   ├── ffmpeg_stream.cpp/.h    ← FFmpeg decoding and OpenCV conversion
   ├── yolo_infefrence.cpp/h   ← ONNXRuntime model handling
   ├── embedded_nr.cpp/.h      ← Signal processing & optimization
```
## 🔧 Setup & Build Instructions
Tested on: **Raspberry Pi 5**, Ubuntu 22.04 (64-bit)

### 1. Clone the Repo
```bash
git clone https://github.com/yashasudupa/multiscale-signal-analysis.git
cd multiscale-signal-analysis
```

### 2. Set Executable Permission
```bash
chmod +x build.sh
```

### 3. Install System Dependencies
The build.sh script will read from requirements.txt and install everything via apt.
```bash
./build.sh <video_stream_url> <model.onnx>
```

The build.sh script will read from requirements.txt and install everything via apt.
```bash
./build.sh rtsp://192.168.1.10:8554/stream yolov5n.onnx
```

### 4. Runtime Logs
```xml
[INFO] Initializing video stream...
[INFO] FFmpeg initialized: rtsp://192.168.1.10:8554/stream
[INFO] ONNX model loaded: yolov5n.onnx
[INFO] Capturing and decoding frame...
[DEBUG] Frame 001: Detected 3 objects
[INFO] Inference complete. Frame saved to ./output/frame_001.jpg
```

## Project Highlights
✅ Camera Interface using FFmpeg
✅ ONNX Runtime-based YOLO Inference
✅ Optimized with C++ and Inline Assembly
✅ Data Compression + Signal Processing Modules
✅ Augmentation-ready Dataset Preparation
✅ Future-ready for FreeRTOS and Frequency Domain Analysis

## Planned Enhancements
Wavelet Transform, FFT, and Power Spectral Density integration
Real-time task prioritization using FreeRTOS
GPU acceleration support for inference (Jetson / Pi with TPU)
