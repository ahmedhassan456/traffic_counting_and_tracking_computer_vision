# Traffic Counting and Tracking

## Overview
This project utilizes computer vision techniques to count and track vehicles in a video stream. It is built using Python and leverages the YOLO (You Only Look Once) algorithm for object detection, along with SORT (Simple Online and Realtime Tracking) for tracking objects across frames.

## Features
- Object detection with YOLOv8
- Object tracking with SORT algorithm
- Filtering of vehicle classes such as cars, buses, trucks, and motorcycles
- Counting vehicles crossing a defined line
- Visual feedback for tracking and counting

## Installation
To run this project, you will need to install the following dependencies:
```bash
pip install opencv-python
pip install cvzone
pip install numpy
pip install ultralytics
