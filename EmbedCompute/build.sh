#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define paths and variables
PROJECT_DIR=$(pwd)  # Current directory
BUILD_DIR="${PROJECT_DIR}/build"
ONNX_RUNTIME_DIR="/home/$USER/onnxruntime"  # Change this if ONNX Runtime is in a different location
VIDEO_STREAM_URL="$1"  # First argument: video stream URL
MODEL_PATH="$2"  # Second argument: path to the ONNX model

# Check for required arguments
if [[ -z "$VIDEO_STREAM_URL" || -z "$MODEL_PATH" ]]; then
    echo "Usage: ./build.sh <video_stream_url> <model.onnx>"
    exit 1
fi

# Step 1: Update and install required dependencies
echo "Installing required packages from requirements.txt..."
if [[ -f "requirements.txt" ]]; then
    while read -r package; do
        sudo apt install -y "$package"
    done < requirements.txt
else
    echo "requirements.txt not found! Please ensure it is present in the project directory."
    exit 1
fi

# Step 2: Download and set up ONNX Runtime (if not already installed)
if [[ ! -d "$ONNX_RUNTIME_DIR" ]]; then
    echo "Downloading ONNX Runtime..."
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-aarch64-1.15.1.tgz -O onnxruntime.tgz
    mkdir -p "$ONNX_RUNTIME_DIR"
    tar -xvf onnxruntime.tgz -C "$ONNX_RUNTIME_DIR" --strip-components=1
    rm onnxruntime.tgz
else
    echo "ONNX Runtime already exists at $ONNX_RUNTIME_DIR."
fi

# Step 3: Create build directory
echo "Creating build directory..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Step 4: Run CMake
echo "Running CMake..."
cmake -DONNX_RUNTIME_DIR="$ONNX_RUNTIME_DIR" ..

# Step 5: Build the project
echo "Building the project..."
make -j$(nproc)

# Step 6: Run the executable
EXECUTABLE="./YOLO_FFmpeg_Inference"
if [[ -f "$EXECUTABLE" ]]; then
    echo "Running the program..."
    $EXECUTABLE "$VIDEO_STREAM_URL" "$MODEL_PATH"
else
    echo "Executable not found! Please check the build process."
    exit 1
fi
