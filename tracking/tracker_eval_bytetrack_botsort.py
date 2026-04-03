#!/usr/bin/env python
"""
Complete Pipeline — RF-DETR Large Model × Both Trackers × All Scenes
==================================================================
"""

import os
import cv2
import numpy as np
import motmetrics as mm
from collections import defaultdict
from pathlib import Path
import csv
import torch
import gc

# GPU pin
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import RF-DETR
from rfdetr import RFDETRSegLarge
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import IterableSimpleNamespace

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Model path - USE THE LARGE MODEL
MODEL_SIZE = "large"
MODEL_PATH = f"./checkpoint_best_ema.pth"
MODEL_NAME = f"RF-DETR-{MODEL_SIZE.capitalize()}"

# All scenes
SCENES = [1, 2, 15, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 61, 63]
VAL_SCENES = [25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37]
TEST_SCENES = [1, 2, 15, 32, 61, 63]

# Trackers
TRACKERS = ["bytetrack", "botsort"]

# Tracking parameters
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
DEVICE = 0

# Output directory
OUTPUT_ROOT = f"./{MODEL_NAME}"

# BotSort hyperparameters
BOTSORT_ARGS = dict(
    tracker_type="botsort",
    track_high_thresh=0.5,
    track_low_thresh=0.1,
    new_track_thresh=0.6,
    track_buffer=30,
    match_thresh=0.8,
    fuse_score=True,
    with_reid=False,
    proximity_thresh=0.5,
    appearance_thresh=0.25,
    gmc_method="sparseOptFlow",
    model=None,
)

# Evaluation IoU threshold
EVAL_IOU_THRESHOLD = 0.5

# Visualization settings
SHOW_BOXES = True
SHOW_LABELS = True
SHOW_TRAILS = True
TRAIL_LENGTH = 30

# Colors
COLOR_PALETTE = [(255, 128, 0), (0, 200, 255), (200, 0, 255), 
                  (255, 255, 0), (0, 255, 128), (255, 0, 128)]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1

# ═══════════════════════════════════════════════════════════════════════


def load_gt_frames(gt_file):
    """Load frame numbers from ground truth file"""
    frames = set()
    with open(gt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                frames.add(int(line.split(',')[0]))
    return sorted(frames)


def load_mot_file(filepath):
    """Load MOT format file into dictionary by frame"""
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            frame = int(parts[0])
            obj_id = int(parts[1])
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            data.setdefault(frame, []).append((obj_id, x, y, w, h))
    return data


def iou_distance(gt_boxes, pred_boxes):
    """Calculate IoU distance matrix"""
    dist = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, (_, gx, gy, gw, gh) in enumerate(gt_boxes):
        for j, (_, px, py, pw, ph) in enumerate(pred_boxes):
            ix1 = max(gx, px)
            iy1 = max(gy, py)
            ix2 = min(gx + gw, px + pw)
            iy2 = min(gy + gh, py + ph)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                dist[i, j] = 1.0
                continue
            union = gw * gh + pw * ph - inter
            dist[i, j] = 1.0 - (inter / union)
    return dist


def draw_tracks(frame, tracks, trail_buf):
    """Draw tracked objects on frame"""
    if tracks is None or len(tracks) == 0:
        return frame
    
    out = frame.copy()
    for row in tracks:
        if len(row) >= 6:
            x1, y1, x2, y2, tid, score = row[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = COLOR_PALETTE[tid % len(COLOR_PALETTE)]
            
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            
            if SHOW_LABELS:
                label = f"ID:{tid} {score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
                cv2.rectangle(out, (x1, max(y1 - th - 4, 0)), 
                             (x1 + tw + 4, max(y1, th + 4)), color, -1)
                cv2.putText(out, label, (x1 + 2, max(y1 - 2, th)), 
                           FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
            
            if SHOW_TRAILS:
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                trail_buf.setdefault(tid, []).append(center)
                if len(trail_buf[tid]) > TRAIL_LENGTH:
                    trail_buf[tid].pop(0)
                
                pts = trail_buf[tid]
                for k in range(1, len(pts)):
                    alpha = k / len(pts)
                    cv2.line(out, pts[k-1], pts[k], 
                            (int(255 * alpha), int(100 * alpha), int(200 * alpha)), 2)
    
    return out


def evaluate_scene(sid, results_file, gt_file):
    """Evaluate a single scene and return metrics"""
    if not os.path.exists(gt_file):
        return None
    if not os.path.exists(results_file):
        return None

    gt_data = load_mot_file(gt_file)
    res_data = load_mot_file(results_file)

    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_data.keys()) | set(res_data.keys()))

    for frame in all_frames:
        gt_objs = gt_data.get(frame, [])
        res_objs = res_data.get(frame, [])
        gt_ids = [o[0] for o in gt_objs]
        res_ids = [o[0] for o in res_objs]

        if gt_objs and res_objs:
            dist = iou_distance(gt_objs, res_objs)
            dist[dist > (1.0 - EVAL_IOU_THRESHOLD)] = np.nan
        elif gt_objs:
            dist = np.full((len(gt_objs), 0), np.nan)
        elif res_objs:
            dist = np.full((0, len(res_objs)), np.nan)
        else:
            dist = np.empty((0, 0))

        acc.update(gt_ids, res_ids, dist)

    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=[
            'num_frames', 'num_unique_objects', 'num_objects',
            'mota', 'motp', 'idf1',
            'num_false_positives', 'num_misses', 'num_switches',
            'mostly_tracked', 'partially_tracked', 'mostly_lost',
            'num_detections',
        ],
        name='tracker'
    )

    total_targets = summary.loc['tracker', 'num_unique_objects']
    mt_count = summary.loc['tracker', 'mostly_tracked']
    pt_count = summary.loc['tracker', 'partially_tracked']
    mt_percentage = (mt_count / total_targets) * 100 if total_targets > 0 else 0.0
    pt_percentage = (pt_count / total_targets) * 100 if total_targets > 0 else 0.0

    return {
        "scene": sid,
        "MOTA": summary.loc['tracker', 'mota'] * 100,
        "IDF1": summary.loc['tracker', 'idf1'] * 100,
        "MT%": mt_percentage,
        "PT%": pt_percentage,
        "FP": int(summary.loc['tracker', 'num_false_positives']),
        "FN": int(summary.loc['tracker', 'num_misses']),
        "IDSW": int(summary.loc['tracker', 'num_switches']),
        "GT_boxes": int(summary.loc['tracker', 'num_objects']),
        "GT_targets": int(total_targets),
        "MT_count": int(mt_count),
        "PT_count": int(pt_count),
    }


def run_bytetrack_and_save(sid, model, output_dir, skip_existing=True):
    """Run ByteTrack inference with RF-DETR, save results and tracked video"""
    video_path = f"{sid}/scene{sid}.mp4"
    gt_file = f"{sid}/gt_scene{sid}.txt"
    result_file = output_dir / "results.txt"
    video_file = output_dir / "tracked_video.mp4"

    # Check if already processed
    if skip_existing and result_file.exists() and video_file.exists():
        print(f"    ⏭️  Already processed, skipping...")
        return result_file, None

    if not os.path.exists(video_path):
        print(f"    ⚠️  Video not found: {video_path}")
        return None, None

    if not os.path.exists(gt_file):
        print(f"    ⚠️  GT file not found: {gt_file}")
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)

    gt_frames = load_gt_frames(gt_file)
    video_frames = sorted(f - 1 for f in gt_frames)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ❌  Cannot open video: {video_path}")
        return None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(
        str(video_file),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W, H)
    )

    import supervision as sv
    tracker = sv.ByteTrack()

    result_lines = []
    trail_buf = {}

    for video_idx in video_frames:
        if video_idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = model.predict(pil_image, threshold=CONFIDENCE_THRESHOLD)
        
        mot_frame = video_idx + 1

        if detections is not None and len(detections.xyxy) > 0:
            sv_detections = sv.Detections(
                xyxy=detections.xyxy,
                confidence=detections.confidence,
                class_id=detections.class_id if detections.class_id is not None else np.zeros(len(detections.xyxy), dtype=int)
            )
            tracked = tracker.update_with_detections(detections=sv_detections)

            if len(tracked) > 0:
                for i, tracker_id in enumerate(tracked.tracker_id):
                    box = tracked.xyxy[i]
                    x, y = int(box[0]), int(box[1])
                    w, h = int(box[2] - box[0]), int(box[3] - box[1])
                    result_lines.append(f"{mot_frame},{tracker_id},{x},{y},{w},{h},-1,-1,-1")
            
            current_tracks = []
            if len(tracked) > 0:
                for i, tid in enumerate(tracked.tracker_id):
                    box = tracked.xyxy[i]
                    current_tracks.append([box[0], box[1], box[2], box[3], tid, tracked.confidence[i]])
        else:
            current_tracks = []

        annotated = draw_tracks(frame, current_tracks, trail_buf)
        cv2.putText(annotated, f"Frame: {video_idx}", (10, 30), FONT, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Tracked: {len(current_tracks)}", (10, 55), FONT, 0.6, (255, 255, 255), 1)

        video_writer.write(annotated)

    cap.release()
    video_writer.release()

    with open(result_file, 'w') as f:
        f.write("\n".join(result_lines))

    print(f"    ✅  Saved video: {video_file}")
    print(f"    ✅  Saved {len(result_lines)} detections to {result_file}")

    return result_file, video_writer


def run_botsort_and_save(sid, model, output_dir, skip_existing=True):
    """Run BotSort inference with RF-DETR, save results and tracked video"""
    video_path = f"{sid}/scene{sid}.mp4"
    gt_file = f"{sid}/gt_scene{sid}.txt"
    result_file = output_dir / "results.txt"
    video_file = output_dir / "tracked_video.mp4"

    # Check if already processed
    if skip_existing and result_file.exists() and video_file.exists():
        print(f"    ⏭️  Already processed, skipping...")
        return result_file, None

    if not os.path.exists(video_path):
        print(f"    ⚠️  Video not found: {video_path}")
        return None, None

    if not os.path.exists(gt_file):
        print(f"    ⚠️  GT file not found: {gt_file}")
        return None, None

    output_dir.mkdir(parents=True, exist_ok=True)

    gt_frames = load_gt_frames(gt_file)
    video_frames = sorted(f - 1 for f in gt_frames)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ❌  Cannot open video: {video_path}")
        return None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    video_writer = cv2.VideoWriter(
        str(video_file),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W, H)
    )

    tracker = BOTSORT(IterableSimpleNamespace(**BOTSORT_ARGS), frame_rate=fps)
    trail_buf = {}

    result_lines = []

    for vid_idx in video_frames:
        if vid_idx >= total:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, vid_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections = model.predict(pil_image, threshold=CONFIDENCE_THRESHOLD)
        
        mot_frame = vid_idx + 1

        tracks = None
        if detections is not None and len(detections.xyxy) > 0:
            det_list = []
            for i in range(len(detections.xyxy)):
                x1, y1, x2, y2 = detections.xyxy[i]
                conf = detections.confidence[i] if detections.confidence is not None else 1.0
                det_list.append([x1, y1, x2, y2, conf])
            
            tracks = [[x1, y1, x2, y2, i, conf] for i, (x1, y1, x2, y2, conf) in enumerate(det_list)]
            
            for row in tracks:
                x1, y1, x2, y2, tid, score = row[:6]
                result_lines.append(
                    f"{mot_frame},{int(tid)},{int(x1)},{int(y1)},{int(x2 - x1)},{int(y2 - y1)},-1,-1,-1"
                )

        annotated = draw_tracks(frame, tracks, trail_buf)
        cv2.putText(annotated, f"Frame: {vid_idx}", (10, 30), FONT, 0.6, (255, 255, 255), 1)
        cv2.putText(annotated, f"Tracked: {len(tracks) if tracks else 0}", (10, 55), FONT, 0.6, (255, 255, 255), 1)

        video_writer.write(annotated)

    cap.release()
    video_writer.release()

    with open(result_file, 'w') as f:
        f.write("\n".join(result_lines))

    print(f"    ✅  Saved video: {video_file}")
    print(f"    ✅  Saved {len(result_lines)} detections to {result_file}")

    return result_file, video_writer


def save_summary(scene_results, output_file, scene_list, split_name):
    """Save summary of metrics for a set of scenes - using ASCII only"""
    if not scene_results:
        return

    lines = []
    lines.append("=" * 80)
    lines.append(f"  {split_name.upper()} VIDEOS SUMMARY — {MODEL_NAME}")
    lines.append("=" * 80)
    lines.append(
        f"  {'Scene':>6}  {'MOTA':>8}  {'IDF1':>8}  {'MT%':>7}  {'PT%':>7}  "
        f"{'FP':>7}  {'FN':>7}  {'IDSW':>6}"
    )
    lines.append("-" * 80)

    for r in scene_results:
        lines.append(
            f"  {r['scene']:>6}  {r['MOTA']:>7.2f}%  {r['IDF1']:>7.2f}%  "
            f"{r['MT%']:>6.1f}%  {r['PT%']:>6.1f}%  "
            f"{r['FP']:>7}  {r['FN']:>7}  {r['IDSW']:>6}"
        )

    if scene_results:
        lines.append("-" * 80)
        lines.append(
            f"  {'AVERAGE':>6}  {np.mean([r['MOTA'] for r in scene_results]):>7.2f}%  "
            f"{np.mean([r['IDF1'] for r in scene_results]):>7.2f}%  "
            f"{np.mean([r['MT%'] for r in scene_results]):>6.1f}%  "
            f"{np.mean([r['PT%'] for r in scene_results]):>6.1f}%  "
            f"{np.mean([r['FP'] for r in scene_results]):>7.0f}  "
            f"{np.mean([r['FN'] for r in scene_results]):>7.0f}  "
            f"{np.mean([r['IDSW'] for r in scene_results]):>6.0f}"
        )
    lines.append("=" * 80)

    output = "\n".join(lines)
    print(output)

    # Write with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output + "\n")


def main():
    print(f"\n{'='*70}")
    print(f"  COMPLETE PIPELINE — {MODEL_NAME}")
    print(f"  Model path: {MODEL_PATH}")
    print(f"{'='*70}")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return

    # Create output directory structure
    output_dir = Path(OUTPUT_ROOT)
    output_dir.mkdir(exist_ok=True)

    # Load RF-DETR model
    print(f"\nLoading RF-DETR model...")
    try:
        model = RFDETRSegLarge(pretrain_weights=MODEL_PATH)
        model.optimize_for_inference()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading with pretrain_weights: {e}")
        print("Trying to load model first then load weights...")
        model = RFDETRSegLarge()
        checkpoint = torch.load(MODEL_PATH, map_location='cuda')
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        model.optimize_for_inference()
        print("Model loaded with strict=False")

    # Store all results for final summary
    all_results = []

    # Process each tracker
    for tracker_name in TRACKERS:
        print(f"\n{'='*70}")
        print(f"  Processing: {tracker_name.upper()} — {MODEL_NAME}")
        print(f"{'='*70}")

        tracker_dir = output_dir / tracker_name
        tracker_dir.mkdir(exist_ok=True)

        for split_name, scenes in [("validation", VAL_SCENES), ("test", TEST_SCENES)]:
            split_dir = tracker_dir / split_name
            split_dir.mkdir(exist_ok=True)

            print(f"\n  {split_name.upper()} SCENES")
            print(f"  {'-'*50}")

            scene_results = []

            for sid in scenes:
                print(f"\n    Scene {sid}")

                scene_dir = split_dir / f"scene{sid}"
                
                # Skip if already processed (check for results file)
                result_file = scene_dir / "results.txt"
                if result_file.exists() and (scene_dir / "tracked_video.mp4").exists():
                    print(f"    ⏭️  Already processed, loading existing results...")
                    # Still evaluate to get metrics
                    gt_file = f"{sid}/gt_scene{sid}.txt"
                    result = evaluate_scene(sid, result_file, gt_file)
                    if result:
                        scene_results.append(result)
                        print(f"      📊 MOTA: {result['MOTA']:.2f}%, IDF1: {result['IDF1']:.2f}%")
                    continue

                if tracker_name == "bytetrack":
                    result_file, _ = run_bytetrack_and_save(sid, model, scene_dir, skip_existing=False)
                else:
                    result_file, _ = run_botsort_and_save(sid, model, scene_dir, skip_existing=False)

                if result_file and result_file.exists():
                    gt_file = f"{sid}/gt_scene{sid}.txt"
                    result = evaluate_scene(sid, result_file, gt_file)

                    if result:
                        scene_results.append(result)
                        print(f"      📊 MOTA: {result['MOTA']:.2f}%, IDF1: {result['IDF1']:.2f}%")

            if scene_results:
                summary_file = tracker_dir / f"summary_{split_name}.txt"
                save_summary(scene_results, summary_file, scenes, split_name)

                csv_file = tracker_dir / f"{split_name}_details.csv"
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=scene_results[0].keys())
                    writer.writeheader()
                    writer.writerows(scene_results)
                print(f"\n  Saved detailed CSV to: {csv_file}")

                all_results.extend(scene_results)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Create master summary
    if all_results:
        master_file = output_dir / "master_summary.txt"
        with open(master_file, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"  MASTER SUMMARY — {MODEL_NAME}\n")
            f.write(f"{'='*70}\n\n")

            for tracker_name in TRACKERS:
                tracker_dir = output_dir / tracker_name
                f.write(f"\n{tracker_name.upper()} RESULTS:\n")
                f.write(f"{'-'*50}\n")

                for split_name in ["validation", "test"]:
                    summary_file = tracker_dir / f"summary_{split_name}.txt"
                    if summary_file.exists():
                        with open(summary_file, 'r', encoding='utf-8') as sf:
                            f.write(sf.read())
                            f.write("\n\n")

        print(f"\n  Master summary saved to: {master_file}")

    print(f"\n{'='*70}")
    print(f"  COMPLETE PIPELINE FINISHED!")
    print(f"  Results saved in: {OUTPUT_ROOT}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
