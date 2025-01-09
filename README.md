# Micro model inference

## Overview

This project involves interfacing the external camera of the Raspberry Pi 5 using FFmpeg, processing image data with OpenCV, and preparing datasets for object detection using YOLOv5. The project includes data annotations, image augmentations, and outlines future work for optimization.

## Achievements

1. **Camera Interface with FFmpeg**  
   Successfully interfaced the camera in the Raspberry Pi 5 with FFmpeg. The camera stream was captured and processed to convert frames into a format compatible with OpenCV for further analysis and manipulation.

2. **Data Annotations and Conversion**  
   Developed a system for annotating data and converting the annotations from XML format into a YOLOv5-compatible format.

3. **Image Augmentations**  
   Implemented various image augmentation techniques to enhance the training dataset. This process helps improve the model's robustness and generalization by artificially increasing the diversity of the training data.

4. **Fixed and Floating Point Arithmetic**  
  Implemented floating point arithmetic to optimize computational efficiency and precision in processing tasks.

5. **Data Compression**  
  Implemented techniques for data compression to reduce storage requirements and improve the speed of data transfer and processing.

6. **Assembly and C++ Programming**  
  Optimize the existing Python scripts by converting them into Assembly and C++ programming to minimize overhead and enhance performance.

## Future Work
**RTOS-Optimized Object Detection with Asynchronous Post-Processing
- Offload non-critical tasks to a lower priority to improve detection performance.
## Conclusion

The project successfully establishes a framework for image processing and object detection of night vision images using the Raspberry Pi 5 and FFmpeg. Optimized performance and increased the efficiency of the pipeline through advanced programming techniques. Working on real-time processing using FreeRTOS for efficient image processing.
