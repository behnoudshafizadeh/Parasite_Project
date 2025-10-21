# Parasite Detection and Tracking Project

⚠️ This repository is shared for peer-review purposes only.
All rights reserved. No reproduction, distribution, or modification is permitted without the author's explicit permission.

## Project Overview
This project implements parasite detection and tracking using YOLO-based object detection with k-fold cross validation, combined with various tracking algorithms (ByteTrack, DeepSORT, BoT-SORT).

## Files Structure
```
├── Track_results-SpecificFrames.py      # ByteTrack evaluation script
├── Yolo_k-fold_segmentation.py         # YOLO training with k-fold validation
├── deepsortTrack_results-SpecificFrames.py  # DeepSORT evaluation script
├── custom-botsort.yaml                 # BoT-SORT configuration
├── custom-bytetrack.yaml               # ByteTrack configuration
├── custom-deepsort.yaml                # DeepSORT configuration
```

## Requirements
- Python 3.x
- PyTorch
- OpenCV
- Ultralytics YOLO
- NumPy
- ByteTrack
- DeepSORT
- BoT-SORT

## Scripts Description

### 1. Yolo_k-fold_segmentation.py
- Implements YOLO-based parasite detection
- Uses k-fold cross validation for robust model training
- Handles dataset splitting and validation

### 2. Track_results-SpecificFrames.py
- Implements ByteTrack algorithm for parasite tracking
- Evaluates tracking performance on specific video frames
- Generates tracking metrics and visualizations

### 3. deepsortTrack_results-SpecificFrames.py
- Implements DeepSORT tracking algorithm
- Processes video frames for parasite tracking
- Provides tracking results and performance analysis

## Configuration Files
Three YAML files provide configuration settings for different tracking algorithms:
- `custom-bytetrack.yaml`: ByteTrack parameters
- `custom-deepsort.yaml`: DeepSORT parameters
- `custom-botsort.yaml`: BoT-SORT parameters

## Usage
1. Training YOLO model with k-fold validation:
```bash
python Yolo_k-fold_segmentation.py
```

2. Running tracking evaluation:
```bash
# For ByteTrack
python Track_results-SpecificFrames.py

# For DeepSORT
python deepsortTrack_results-SpecificFrames.py
```

## License
© 2025 All rights reserved. This project is protected by copyright law.
No part of this project may be reproduced, distributed, or modified without explicit permission from the author.

## Contact
For inquiries about this project, please contact:
behnoudsahfiezadehkenari@gaslini.org, shayanalvansazyazdi@gmail.com, rosella.tro@unige.it

