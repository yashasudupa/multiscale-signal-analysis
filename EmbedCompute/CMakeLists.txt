cmake_minimum_required(VERSION 3.16)
project(YOLO_FFmpeg_Inference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# Specify the ONNX Runtime installation path (update this as per your setup)
set(ONNX_RUNTIME_DIR /path/to/onnxruntime)

# Specify FFmpeg installation paths (update these as per your setup)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libswscale libavutil)

# Specify OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${ONNX_RUNTIME_DIR}/include
    ${FFMPEG_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Link directories
link_directories(
    ${ONNX_RUNTIME_DIR}/lib
    ${FFMPEG_LIBRARY_DIRS}
)

# Source files
set(SOURCES
    YOLOInference.cpp
    FFmpegStream.cpp
    main.cpp  # Add your main entry point
)

# Add executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
    onnxruntime  # Ensure ONNX Runtime is linked
)

# Install target (optional)
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
