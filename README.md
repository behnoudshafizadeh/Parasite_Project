# Automated Tracking of the Brown Algal Parasite *Eurychasma dicksonii* with Deep Learning

> **Notice:** This work is **currently under revision and has not been
> published yet**.\
> The repository is being actively updated alongside the manuscript
> revision process.

## Overview

This repository provides an end-to-end deep learning framework for
**segmentation, multi-object tracking, and biological dynamics
analysis** of *Eurychasma dicksonii* in time-lapse microscopy videos.

The project benchmarks **five segmentation architectures**: - YOLOv8 -
YOLOv11 - YOLOv8-SwinT - RF-DETR Nano - RF-DETR Large

The best segmentation model is then integrated with: - ByteTrack -
BoT-SORT - DeepSORT

to analyze parasite trajectories and quantify **parasite area expansion
over time**.

------------------------------------------------------------------------

## Highlights

-   Benchmark of **5 segmentation architectures**
-   **RF-DETR Large** achieved the best segmentation performance
-   **ByteTrack** delivered the most robust tracking
-   Automated **single-parasite area extraction**
-   **PELT-based temporal phase detection**
-   Excel export for each tracked parasite
-   Biological **area expansion dynamics** analysis

------------------------------------------------------------------------

## Repository Structure

``` text
Parasite_Project/
├── segmentation/
├── tracking/
├── feature_analysis/
├── outputs/
├── images/
```

------------------------------------------------------------------------

## Code Guide by Pipeline Stage

This section helps readers quickly understand **which file to run at
each stage**.

### 1) Segmentation Training

Use one of these files:

``` text
segmentation/rfdetr_large.py
segmentation/rfdetr_nano.py
segmentation/yolov8_swint.py
```

### 2) Tracking Evaluation

Use:

``` text
tracking/tracker_eval_bytetrack_botsort.py
tracking/tracker_eval_deepsort.py
```

### 3) Area Feature Extraction

After tracking, extract **Area time-series per parasite**:

``` text
feature_analysis/extract_shape_features.py
```

> In this project, the **Area feature is the main biological descriptor
> used for downstream analysis**.

### 4) Temporal Phase Detection

Run PELT on the Area signal:

``` text
feature_analysis/pelt_phase_detection.py
```

This detects: - expansion onset - stable growth - shrinkage - transition
points

------------------------------------------------------------------------

## Best Results

  Module         Best Model                  Metric
  -------------- --------------------------- ---------------
  Segmentation   RF-DETR Large               mAP50 = 74.0%
  Tracking       RF-DETR Large + ByteTrack   MOTA = 66.08%
  Identity       RF-DETR Large + ByteTrack   IDF1 = 75.47%

------------------------------------------------------------------------

## Workflow

1.  Data acquisition and frame extraction
2.  Segmentation benchmarking
3.  Multi-object tracking evaluation
4.  Area extraction from masks
5.  PELT temporal phase detection
6.  Biological interpretation of area dynamics

------------------------------------------------------------------------

## Biological Analysis (Area Only)

The downstream biological analysis in this repository focuses on:

-   **Area (px²)**
-   smoothed Area signal
-   Area velocity (**dA/dt**)
-   expansion onset
-   phase boundaries
-   stable vs shrinkage periods

This makes the repository easier to reproduce for biological studies
focused on **parasite growth kinetics**.

------------------------------------------------------------------------

## Contributors & Collaboration

This project was developed with valuable scientific and technical
contributions from:

-   **Rosella Tro** --- rosella.tro@unige.it
-   **Shayan Alvansazyazdi** --- shayanalvansazyazdi@gaslini.org

We welcome collaboration from researchers working in: - microscopy AI -
parasite tracking - time-series biological analysis - deep learning for
life sciences

If you would like to **contribute to this project, extend the pipeline,
or collaborate on related research**, please feel free to contact the
contributors above.
