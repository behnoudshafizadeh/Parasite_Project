"""
Parasite Shape Feature Extraction — RF-DETR Small + ByteTrack
============================================================
Extracts shape and development features from segmented parasite
masks across video frames using RF-DETR Small segmentation model.

Features extracted (shape-change focused):
  Development descriptors:
    - Area                  (px²)
    - Perimeter             (px)
    - Equivalent_Diameter   (px)
    - Compactness           = 4π·Area / Perimeter²  (1=circle, →0=irregular)

  Morphology descriptors:
    - Eccentricity          (0=circle, 1=line)
    - Major_Axis_Length     (px)
    - Minor_Axis_Length     (px)
    - Aspect_Ratio          = Major / Minor
    - Solidity              (area / convex_hull_area — smoothness)
    - Convex_Area           (px²)
    - Extent                (area / bounding_box_area)

Each parasite gets its own Excel file:
    <output_folder>/parasite_<ID>.xlsx

Usage:
    python extracting_features_rfdetr.py
"""

import os
# ── GPU pin — MUST be before ALL other imports ──────────────────────────
# Edit the number below to select your GPU (0, 1, 2, 3 ...)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# ────────────────────────────────────────────────────────────────────────

import gc
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.measure import regionprops, label as sk_label

from rfdetr import RFDETRSegSmall
import supervision as sv


# ═══════════════════════════════════════════════════════════════════════
#  CONFIG  ← only edit this section
# ═══════════════════════════════════════════════════════════════════════

# ── Input ───────────────────────────────────────────────────────────────
VIDEO_PATH  = "test_videos/Folder7_non-abortive.mp4"
CHECKPOINT  = "Small_checkpoint_best_total.pth"

# ── Model class — match to your checkpoint ──────────────────────────────
#   "small" → RFDETRSegSmall
#   "small" → RFDETRSegSmall
MODEL_CLASS = RFDETRSegSmall

# ── Output ──────────────────────────────────────────────────────────────
OUTPUT_FOLDER     = "Folder7_non-abortive_parasite_features"   # one xlsx per parasite saved here
OUTPUT_VIDEO_PATH = "Folder7_non-abortive_parasite_features/feature_tracking.mp4"  # annotated video (set None to skip)

# ── Inference ───────────────────────────────────────────────────────────
GPU        = 3
CONFIDENCE = 0.5

# ByteTrack — no extra config needed, supervision handles defaults

# ═══════════════════════════════════════════════════════════════════════

# GPU already pinned at top of file
import torch



# ───────────────────────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ───────────────────────────────────────────────────────────────────────

def extract_shape_features(binary_mask, frame_idx, track_id):
    """
    Extract shape-change features from a binary mask.
    Returns a dict or None if mask has no valid region.

    Development descriptors:
        Area, Perimeter, Equivalent_Diameter, Compactness

    Morphology descriptors:
        Eccentricity, Major_Axis_Length, Minor_Axis_Length,
        Aspect_Ratio, Solidity, Convex_Area, Extent
    """
    # skimage regionprops needs a labelled integer mask
    labelled = sk_label(binary_mask.astype(np.uint8))
    props    = regionprops(labelled)

    if not props:
        return None

    # Take the largest region (main parasite body)
    prop = max(props, key=lambda p: p.area)

    area      = prop.area
    perimeter = prop.perimeter if prop.perimeter > 0 else 1e-9
    compactness = (4 * np.pi * area) / (perimeter ** 2)   # 1=perfect circle

    major = prop.major_axis_length
    minor = prop.minor_axis_length if prop.minor_axis_length > 0 else 1e-9
    aspect_ratio = major / minor

    return {
        # ── Identifiers ───────────────────────────────────────────────
        "Frame_Index"       : int(frame_idx),
        "Track_ID"          : int(track_id),

        # ── Development descriptors ───────────────────────────────────
        # Area over time → used to compute slope, dA/dt, sigmoid fitting
        "Area"              : float(area),
        # Perimeter — changes as parasite expands/ruptures
        "Perimeter"         : float(perimeter),
        # Equivalent diameter of a circle with same area
        "Equivalent_Diameter": float(prop.equivalent_diameter),
        # Compactness: 1=circle, decreases as shape becomes irregular
        # Useful to detect transition from intracellular → expanded sporangia
        "Compactness"       : float(compactness),

        # ── Morphology descriptors ────────────────────────────────────
        # Eccentricity: 0=circle, 1=line — captures elongation during development
        "Eccentricity"      : float(prop.eccentricity),
        # Major axis — tracks directional growth
        "Major_Axis_Length" : float(major),
        # Minor axis — tracks lateral growth
        "Minor_Axis_Length" : float(prop.minor_axis_length),
        # Aspect ratio = Major / Minor — shape elongation ratio
        "Aspect_Ratio"      : float(aspect_ratio),
        # Solidity = area / convex_hull_area — detects surface irregularity
        # Drops when parasite membrane ruptures or becomes jagged
        "Solidity"          : float(prop.solidity),
        # Convex area — tracks overall occupied space including concavities
        "Convex_Area"       : float(prop.convex_area),
        # Extent = area / bounding_box_area — rectangularity
        "Extent"            : float(prop.extent),
    }


# ───────────────────────────────────────────────────────────────────────
#  MAIN
# ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print(f"\n{'═'*62}")
    print(f"  Parasite Shape Feature Extraction — RF-DETR Small")
    print(f"{'═'*62}")
    print(f"  Video      : {VIDEO_PATH}")
    print(f"  Checkpoint : {CHECKPOINT}")
    print(f"  Output     : {OUTPUT_FOLDER}/")
    print(f"{'═'*62}\n")

    # ── Load model ────────────────────────────────────────────────────
    if not os.path.exists(CHECKPOINT):
        print(f"❌  Checkpoint not found: {CHECKPOINT}")
        return

    print("🔧  Loading RF-DETR Small ...")
    model = RFDETRSegSmall(pretrain_weights=CHECKPOINT)
    model.optimize_for_inference()
    print("✅  Model ready\n")

    # ── Open video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌  Cannot open video: {VIDEO_PATH}")
        return

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹  {total} frames | {fps:.1f} FPS | {W}×{H}\n")

    # ── Setup tracker ─────────────────────────────────────────────────
    tracker = sv.ByteTrack()

    # ── Setup video writer ────────────────────────────────────────────
    writer = None
    if OUTPUT_VIDEO_PATH:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (W, H))

    # ── Data store: {track_id: [feature_dict, ...]} ───────────────────
    all_data = {}   # {track_id: list of feature dicts}

    print(f"🚀  Extracting features ...\n")

    frame_idx = 0
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # ── Detect ────────────────────────────────────────────────────
        pil  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        dets = model.predict(pil, threshold=CONFIDENCE)

        # ── Track ─────────────────────────────────────────────────────
        tracked = tracker.update_with_detections(detections=dets)

        # ── Extract masks and match to track IDs ──────────────────────
        annotated = frame_bgr.copy()

        if (tracked.tracker_id is not None and len(tracked.tracker_id) > 0
                and tracked.mask is not None and len(tracked.mask) > 0):

            for i, track_id in enumerate(tracked.tracker_id):
                track_id = int(track_id)

                # Get binary mask for this tracked detection
                raw_mask = tracked.mask[i]   # bool array H×W
                if raw_mask.shape != (H, W):
                    raw_mask = cv2.resize(
                        raw_mask.astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)

                binary_mask = raw_mask.astype(np.uint8)

                # Extract shape features
                feats = extract_shape_features(binary_mask, frame_idx, track_id)
                if feats is None:
                    continue

                if track_id not in all_data:
                    all_data[track_id] = []
                all_data[track_id].append(feats)

                # ── Annotate frame ─────────────────────────────────────
                box   = tracked.xyxy[i]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                color = [(255,128,0),(0,200,255),(200,0,255),
                         (255,255,0),(0,255,128),(255,0,128)][track_id % 6]

                # Draw mask overlay
                overlay           = annotated.copy()
                overlay[raw_mask] = color
                annotated         = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

                # Draw box and label
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
                label = (f"ID:{track_id}  "
                         f"A:{feats['Area']:.0f}  "
                         f"Ecc:{feats['Eccentricity']:.2f}")
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated,
                              (x1, max(y1-th-6,0)),
                              (x1+tw+4, max(y1, th+6)), color, -1)
                cv2.putText(annotated, label,
                            (x1+2, max(y1-3, th)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255,255,255), 1)

        # Frame info
        cv2.putText(annotated,
                    f"Frame {frame_idx}  |  Tracked: {len(all_data)} parasites",
                    (8, H-12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200,200,200), 1)

        if writer:
            writer.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"   Frame {frame_idx:5d}/{total}  |  "
                  f"Active parasites: {len(all_data)}")

    cap.release()
    if writer:
        writer.release()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Save one Excel per parasite ───────────────────────────────────
    print(f"\n💾  Saving Excel files ...")
    for track_id, records in sorted(all_data.items()):
        df       = pd.DataFrame(records)
        filename = os.path.join(OUTPUT_FOLDER, f"parasite_{track_id}.xlsx")
        df.to_excel(filename, index=False)
        print(f"   parasite_{track_id}.xlsx  —  {len(df)} frames")

    print(f"\n{'═'*62}")
    print(f"  ✅  Done!")
    print(f"  📁  {len(all_data)} Excel files saved → {OUTPUT_FOLDER}/")
    if OUTPUT_VIDEO_PATH:
        print(f"  🎥  Annotated video → {OUTPUT_VIDEO_PATH}")
    print(f"{'═'*62}\n")

    # ── Print feature summary ─────────────────────────────────────────
    print("  FEATURE COLUMNS SAVED PER PARASITE:")
    print("  ─────────────────────────────────────────────────────")
    feature_info = [
        ("Frame_Index",        "frame number (for time axis)"),
        ("Track_ID",           "parasite identity"),
        ("Area",               "px² — key development signal (sigmoid over time)"),
        ("Perimeter",          "px — boundary length"),
        ("Equivalent_Diameter","px — diameter of same-area circle"),
        ("Compactness",        "4πA/P² — 1=circle, drops at rupture/irregular growth"),
        ("Eccentricity",       "0=circle, 1=line — elongation"),
        ("Major_Axis_Length",  "px — directional growth axis"),
        ("Minor_Axis_Length",  "px — lateral growth axis"),
        ("Aspect_Ratio",       "Major/Minor — elongation ratio"),
        ("Solidity",           "A/ConvexA — surface smoothness, drops at rupture"),
        ("Convex_Area",        "px² — convex hull area"),
        ("Extent",             "A/BBoxA — rectangularity"),
    ]
    for col, desc in feature_info:
        print(f"  {col:<22} : {desc}")
    print()


if __name__ == "__main__":
    main()
