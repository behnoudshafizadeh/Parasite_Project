# Automated Tracking of the Brown Algal Parasite *Eurychasma dicksonii* with Deep Learning

## Overview
This project provides an automated deep learning framework for segmentation, tracking, and biological analysis of *Eurychasma dicksonii* in time-lapse microscopy videos.

The framework benchmarks multiple segmentation architectures including YOLOv8, YOLOv11, YOLOv8-SwinT, RF-DETR Nano, and RF-DETR Large, followed by multi-object tracking using ByteTrack, BoT-SORT, and DeepSORT.

## Highlights
- 5 segmentation architectures benchmarked
- RF-DETR Large best segmentation performance
- ByteTrack best tracking robustness
- automated parasite shape feature extraction
- PELT-based expansion phase detection
- export to Excel for each parasite

## Repository Structure
```text
segmentation/
tracking/
feature_analysis/
outputs/
paper/
