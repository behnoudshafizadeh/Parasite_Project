# Parasite Detection and Tracking Project

⚠️ This repository is shared for peer-review purposes only.
All rights reserved. No reproduction, distribution, or modification is permitted without the author's explicit permission.

## Project Overview
This project implements parasite detection and tracking using YOLO-based object detection combined with various tracking algorithms. The system is designed to detect and track parasites in video sequences with high accuracy.

## Features
- YOLO-based parasite detection with k-fold cross validation
- Multiple tracking algorithm implementations:
  - ByteTrack
  - DeepSORT
  - BoT-SORT
- Performance evaluation using MOT metrics
- Support for specific frame analysis

## Project Structure
```
Parasite Project/
├── shayan-yolov11.py           # YOLO training script
├── Track_results-SpecificFrames.py      # ByteTrack evaluation
├── deepsortTrack_results-SpecificFrames.py  # DeepSORT evaluation
├── Configuration/
│   ├── custom-botsort.yaml     # BoT-SORT config
│   ├── custom-bytetrack.yaml   # ByteTrack config
│   └── custom-deepsort.yaml    # DeepSORT config
├── Labels/
│   └── Validation Images/Labels
├── Videos/
    ├── Test/
    ├── Train/
    └── Validation/
```

## Requirements
- Python 3.x
- PyTorch
- OpenCV
- Ultralytics YOLO
- NumPy
- Tracking dependencies (ByteTrack, DeepSORT, BoT-SORT)

## Usage
1. Training YOLO model:
```bash
python shayan-yolov11.py
```

2. Running tracking evaluation:
```bash
python Track_results-SpecificFrames.py  # For ByteTrack
python deepsortTrack_results-SpecificFrames.py  # For DeepSORT
```

## Configuration
Tracker configurations can be modified in the respective YAML files:
- `custom-bytetrack.yaml`
- `custom-deepsort.yaml`
- `custom-botsort.yaml`

## Citation
If you use this project in your research, please cite:
[Add citation information]

## License
© 2025 All rights reserved. This project is protected by copyright law.
No part of this project may be reproduced, distributed, or modified without explicit permission from the author.

## Contact
[Add contact information]
