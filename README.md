# Multi scale signal analysis

![image](https://github.com/user-attachments/assets/2d81388b-70e7-4195-b6e8-3e21e4bc02c6)

## Overview

This project involves interfacing an external camera with the Raspberry Pi 5 using FFmpeg, processing image data, and preparing datasets for object detection. It includes implementing data compression and optimizing night vision wildlife image detection using Assembly and C++. The project also explores future enhancements, such as integrating frequency domain analysis and real-time task scheduling with FreeRTOS.

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
1. Integrate Wavelet Transform, Fourier Transform (FFT), and Power Spectral Density (PSD) to improve detection capabilities.
   
2. Real-time task scheduling with FreeRTOS to offload non-critical tasks to lower priorities, enhancing detection performance.
