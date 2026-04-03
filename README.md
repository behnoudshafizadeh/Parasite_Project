# Automated Tracking of the Brown Algal Parasite *Eurychasma dicksonii* with Deep Learning

> **Notice:** This work is **currently under revision and has not been
> published yet**.\
> The repository is being actively updated alongside the manuscript
> revision process.

## Overview

This repository provides an end-to-end deep learning framework for
**segmentation, multi-object tracking, and biological dynamics
analysis** of *Eurychasma dicksonii* in time-lapse microscopy videos.

The study includes both the **earlier YOLO-based pipeline** and the
**new transformer-based RF-DETR pipeline**.

### Segmentation models explored

-   YOLOv8
-   YOLOv11
-   YOLOv8-SwinT
-   RF-DETR Nano
-   RF-DETR Large

### Tracking models integrated

-   ByteTrack
-   BoT-SORT
-   DeepSORT

The final biological analysis focuses on **parasite area expansion over
time**.

------------------------------------------------------------------------

## Highlights

-   Benchmark of **5 segmentation architectures**
-   Includes **legacy YOLO pipeline + new RF-DETR pipeline**
-   **RF-DETR Large** achieved the best segmentation performance
-   **ByteTrack** delivered the most robust tracking
-   Automated **single-parasite area extraction**
-   **PELT-based temporal phase detection**
-   Biological **area expansion dynamics** analysis

------------------------------------------------------------------------

## Repository Structure

``` text
Parasite_Project/
├── legacy_yolo_pipeline/
├── segmentation/
├── tracking/
├── feature_analysis/
├── outputs/
```

------------------------------------------------------------------------

## Code Guide by Pipeline Stage

### 1) Legacy YOLOv8 / YOLOv11 Training + Tracking

These are your **original GitHub files already in the repository**:

``` text
Segmentation_K-fold.py
Track_results-SpecificFrames.py
deepsortTrack_results-SpecificFrames.py
custom-bytetrack.yaml
custom-botsort.yaml
custom-deepsort.yaml
```

Use these files if readers want to reproduce the **first YOLO-based
experiments**, where: - YOLOv8 / YOLOv11 were trained - evaluated with
K-fold validation - integrated with ByteTrack / BoT-SORT / DeepSORT

This is important because these files represent the **first benchmark
stage of the project**.

------------------------------------------------------------------------

### 2) New Transformer Segmentation Training

Use the newer files for the updated paper experiments:

``` text
segmentation/rfdetr_large.py
segmentation/rfdetr_nano.py
segmentation/yolov8_swint.py
```

------------------------------------------------------------------------

### 3) New Tracking Evaluation

Use:

``` text
tracking/tracker_eval_bytetrack_botsort.py
tracking/tracker_eval_deepsort.py
```

------------------------------------------------------------------------

### 4) Area Feature Extraction

After tracking, extract **Area time-series per parasite**:

``` text
feature_analysis/extract_shape_features.py
```

> The **Area feature is the main biological descriptor used for
> downstream analysis**.

------------------------------------------------------------------------

### 5) Temporal Phase Detection

Run PELT on the Area signal:

``` text
feature_analysis/pelt_phase_detection.py
```

This detects: - expansion onset - stable growth - shrinkage - transition
points

------------------------------------------------------------------------

## Contributors & Collaboration

This project was developed with valuable scientific and technical
contributions from:

-   **Rosella Tro** --- rosella.tro@unige.it
-   **Shayan Alvansazyazdi** --- shayanalvansazyazdi@gaslini.org

If you would like to **contribute to this project, extend the pipeline,
or collaborate on related research**, please feel free to contact the
contributors above.

------------------------------------------------------------------------

## Best Results

  Module         Best Model                  Metric
  -------------- --------------------------- ---------------
  Segmentation   RF-DETR Large               mAP50 = 74.0%
  Tracking       RF-DETR Large + ByteTrack   MOTA = 66.08%
  Identity       RF-DETR Large + ByteTrack   IDF1 = 75.47%

------------------------------------------------------------------------

## Workflow

1.  Legacy YOLO segmentation + tracking benchmark
2.  RF-DETR segmentation benchmark
3.  Multi-object tracking evaluation
4.  Area extraction from masks
5.  PELT temporal phase detection
6.  Biological interpretation of area dynamics
