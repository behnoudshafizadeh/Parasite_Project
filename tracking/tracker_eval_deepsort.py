#!/usr/bin/env python
"""
MOT Metrics Evaluation Script — RF-DETR Large Model
===================================================
Evaluates existing tracking results for all trackers (ByteTrack, BoT-SORT, DeepSORT)
and displays/saves comprehensive metrics tables.
"""

import os
import numpy as np
import motmetrics as mm
from pathlib import Path
import csv
from collections import defaultdict
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

MODEL_NAME = "RF-DETR-Large"

# Scenes and splits
SCENES = [1, 2, 15, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 61, 63]
VAL_SCENES = [25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37]
TEST_SCENES = [1, 2, 15, 32, 61, 63]

# Trackers to evaluate
TRACKERS = ["bytetrack", "botsort", "deepsort"]

# Base path where results are stored
BASE_PATH = f"./{MODEL_NAME}"

# IoU threshold for matching
IOU_THRESHOLD = 0.5

# Output files
SUMMARY_FILE = f"./{MODEL_NAME}_evaluation_summary.txt"
CSV_FILE = f"./{MODEL_NAME}_evaluation_summary.csv"

# ═══════════════════════════════════════════════════════════════════════


def load_mot_file(filepath):
    """Load MOT format file into dictionary by frame"""
    data = {}
    if not os.path.exists(filepath):
        return data
   
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
    """Calculate IoU distance matrix between GT and predicted boxes"""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)))
   
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


def evaluate_scene(sid, results_file, gt_file):
    """Evaluate a single scene and return metrics"""
    if not os.path.exists(gt_file):
        return None
    if not os.path.exists(results_file):
        return None
   
    gt_data = load_mot_file(gt_file)
    res_data = load_mot_file(results_file)
   
    if len(gt_data) == 0:
        return None
   
    acc = mm.MOTAccumulator(auto_id=True)
    all_frames = sorted(set(gt_data.keys()) | set(res_data.keys()))
   
    for frame in all_frames:
        gt_objs = gt_data.get(frame, [])
        res_objs = res_data.get(frame, [])
        gt_ids = [o[0] for o in gt_objs]
        res_ids = [o[0] for o in res_objs]
       
        if gt_objs and res_objs:
            dist = iou_distance(gt_objs, res_objs)
            dist[dist > (1.0 - IOU_THRESHOLD)] = np.nan
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
        "MOTP": summary.loc['tracker', 'motp'] * 100,
        "IDF1": summary.loc['tracker', 'idf1'] * 100,
        "MT": mt_percentage,
        "PT": pt_percentage,
        "ML": (1 - (mt_count + pt_count) / total_targets) * 100 if total_targets > 0 else 0,
        "FP": int(summary.loc['tracker', 'num_false_positives']),
        "FN": int(summary.loc['tracker', 'num_misses']),
        "IDSW": int(summary.loc['tracker', 'num_switches']),
        "GT_boxes": int(summary.loc['tracker', 'num_objects']),
        "GT_targets": int(total_targets),
    }


def find_result_files(tracker_name):
    """Find all result files for a given tracker"""
    results = {}
    tracker_dir = Path(BASE_PATH) / tracker_name
   
    if not tracker_dir.exists():
        return results
   
    # Check validation and test scenes
    for split_name, scenes in [("validation", VAL_SCENES), ("test", TEST_SCENES)]:
        split_dir = tracker_dir / split_name
        if split_dir.exists():
            for sid in scenes:
                result_file = split_dir / f"scene{sid}" / "results.txt"
                if result_file.exists():
                    results[(split_name, sid)] = result_file
   
    return results


def compute_aggregate_metrics(all_results):
    """Compute aggregate metrics across all scenes"""
    if not all_results:
        return {}
   
    # Sum raw counts
    total_fp = sum(r['FP'] for r in all_results)
    total_fn = sum(r['FN'] for r in all_results)
    total_idsw = sum(r['IDSW'] for r in all_results)
    total_gt = sum(r['GT_boxes'] for r in all_results)
   
    # Aggregate MOTA
    agg_mota = (1 - (total_fn + total_fp + total_idsw) / total_gt) * 100 if total_gt > 0 else 0
   
    # Weighted averages
    total_gt_targets = sum(r['GT_targets'] for r in all_results)
    if total_gt_targets > 0:
        weighted_mt = sum(r['MT'] * r['GT_targets'] for r in all_results) / total_gt_targets
        weighted_pt = sum(r['PT'] * r['GT_targets'] for r in all_results) / total_gt_targets
        weighted_ml = sum(r['ML'] * r['GT_targets'] for r in all_results) / total_gt_targets
    else:
        weighted_mt = weighted_pt = weighted_ml = 0
   
    # Mean averages
    mean_mota = np.mean([r['MOTA'] for r in all_results])
    mean_idf1 = np.mean([r['IDF1'] for r in all_results])
    mean_mt = np.mean([r['MT'] for r in all_results])
    mean_pt = np.mean([r['PT'] for r in all_results])
    mean_ml = np.mean([r['ML'] for r in all_results])
   
    return {
        "MOTA_agg": agg_mota,
        "MOTA_mean": mean_mota,
        "IDF1_mean": mean_idf1,
        "MT_weighted": weighted_mt,
        "MT_mean": mean_mt,
        "PT_weighted": weighted_pt,
        "PT_mean": mean_pt,
        "ML_weighted": weighted_ml,
        "ML_mean": mean_ml,
        "total_FP": total_fp,
        "total_FN": total_fn,
        "total_IDSW": total_idsw,
        "total_GT": total_gt,
        "num_scenes": len(all_results)
    }


def print_table(results_by_tracker, split_name):
    """Print a formatted table for a specific split"""
    print(f"\n{'='*90}")
    print(f"  {split_name.upper()} SCENES — MOT EVALUATION RESULTS")
    print(f"{'='*90}")
   
    for tracker in TRACKERS:
        if tracker not in results_by_tracker or split_name not in results_by_tracker[tracker]:
            continue
       
        results = results_by_tracker[tracker][split_name]
        if not results:
            continue
       
        print(f"\n  {tracker.upper()}:")
        print(f"  {'-'*85}")
        print(f"  {'Scene':>6}  {'MOTA':>8}  {'IDF1':>8}  {'MT%':>7}  {'PT%':>7}  {'ML%':>7}  "
              f"{'FP':>7}  {'FN':>7}  {'IDSW':>6}")
        print(f"  {'-'*85}")
       
        for r in sorted(results, key=lambda x: x['scene']):
            print(f"  {r['scene']:>6}  {r['MOTA']:>7.2f}%  {r['IDF1']:>7.2f}%  "
                  f"{r['MT']:>6.1f}%  {r['PT']:>6.1f}%  {r['ML']:>6.1f}%  "
                  f"{r['FP']:>7}  {r['FN']:>7}  {r['IDSW']:>6}")
       
        # Print aggregate for this tracker on this split
        agg = compute_aggregate_metrics(results)
        print(f"  {'-'*85}")
        print(f"  {'AGGREGATE':>6}  {agg['MOTA_agg']:>7.2f}%  {agg['IDF1_mean']:>7.2f}%  "
              f"{agg['MT_weighted']:>6.1f}%  {agg['PT_weighted']:>6.1f}%  {agg['ML_weighted']:>6.1f}%  "
              f"{agg['total_FP']:>7}  {agg['total_FN']:>7}  {agg['total_IDSW']:>6}")
        print(f"  {'MEAN':>6}  {agg['MOTA_mean']:>7.2f}%  {agg['IDF1_mean']:>7.2f}%  "
              f"{agg['MT_mean']:>6.1f}%  {agg['PT_mean']:>6.1f}%  {agg['ML_mean']:>6.1f}%  "
              f"{'':>7}  {'':>7}  {'':>6}")


def print_summary_table(results_by_tracker):
    """Print a comprehensive summary table comparing all trackers"""
    print(f"\n{'='*90}")
    print(f"  COMPREHENSIVE SUMMARY — ALL TRACKERS")
    print(f"{'='*90}")
   
    for split_name in ["validation", "test"]:
        print(f"\n\n{'-'*90}")
        print(f"  {split_name.upper()} SET — Aggregated Metrics (Weighted by GT Boxes)")
        print(f"{'-'*90}")
       
        # Header
        print(f"\n  {'Tracker':>12}  {'MOTA':>8}  {'IDF1':>8}  {'MT%':>7}  {'PT%':>7}  {'ML%':>7}  "
              f"{'FP':>8}  {'FN':>8}  {'IDSW':>6}")
        print(f"  {'-'*90}")
       
        for tracker in TRACKERS:
            if tracker not in results_by_tracker or split_name not in results_by_tracker[tracker]:
                continue
           
            results = results_by_tracker[tracker][split_name]
            if not results:
                continue
           
            agg = compute_aggregate_metrics(results)
           
            print(f"  {tracker.upper():>12}  {agg['MOTA_agg']:>7.2f}%  {agg['IDF1_mean']:>7.2f}%  "
                  f"{agg['MT_weighted']:>6.1f}%  {agg['PT_weighted']:>6.1f}%  {agg['ML_weighted']:>6.1f}%  "
                  f"{agg['total_FP']:>8}  {agg['total_FN']:>8}  {agg['total_IDSW']:>6}")
       
        # Add mean across trackers for this split
        print(f"  {'-'*90}")
       
        print(f"\n\n{'-'*90}")
        print(f"  {split_name.upper()} SET — Mean Across Scenes (Unweighted)")
        print(f"{'-'*90}")
       
        print(f"\n  {'Tracker':>12}  {'MOTA':>8}  {'IDF1':>8}  {'MT%':>7}  {'PT%':>7}  {'ML%':>7}")
        print(f"  {'-'*70}")
       
        for tracker in TRACKERS:
            if tracker not in results_by_tracker or split_name not in results_by_tracker[tracker]:
                continue
           
            results = results_by_tracker[tracker][split_name]
            if not results:
                continue
           
            agg = compute_aggregate_metrics(results)
           
            print(f"  {tracker.upper():>12}  {agg['MOTA_mean']:>7.2f}%  {agg['IDF1_mean']:>7.2f}%  "
                  f"{agg['MT_mean']:>6.1f}%  {agg['PT_mean']:>6.1f}%  {agg['ML_mean']:>6.1f}%")


def save_results_to_csv(results_by_tracker):
    """Save all results to CSV file"""
    all_data = []
   
    for tracker in TRACKERS:
        if tracker not in results_by_tracker:
            continue
       
        for split_name in ["validation", "test"]:
            if split_name not in results_by_tracker[tracker]:
                continue
           
            results = results_by_tracker[tracker][split_name]
            for r in results:
                r_copy = r.copy()
                r_copy['tracker'] = tracker
                r_copy['split'] = split_name
                all_data.append(r_copy)
   
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(CSV_FILE, index=False)
        print(f"\n  💾 Detailed results saved to: {CSV_FILE}")


def save_summary_to_file(results_by_tracker):
    """Save comprehensive summary to text file"""
    lines = []
    lines.append("=" * 100)
    lines.append(f"  MOT EVALUATION SUMMARY — {MODEL_NAME}")
    lines.append("=" * 100)
    lines.append("")
   
    for split_name in ["validation", "test"]:
        lines.append(f"\n{'-'*90}")
        lines.append(f"  {split_name.upper()} SET — Aggregated Metrics")
        lines.append(f"{'-'*90}")
       
        lines.append(f"\n  {'Tracker':>12}  {'MOTA':>8}  {'IDF1':>8}  {'MT%':>7}  {'PT%':>7}  {'ML%':>7}  "
                     f"{'FP':>8}  {'FN':>8}  {'IDSW':>6}")
        lines.append(f"  {'-'*90}")
       
        for tracker in TRACKERS:
            if tracker not in results_by_tracker or split_name not in results_by_tracker[tracker]:
                continue
           
            results = results_by_tracker[tracker][split_name]
            if not results:
                continue
           
            agg = compute_aggregate_metrics(results)
           
            lines.append(f"  {tracker.upper():>12}  {agg['MOTA_agg']:>7.2f}%  {agg['IDF1_mean']:>7.2f}%  "
                         f"{agg['MT_weighted']:>6.1f}%  {agg['PT_weighted']:>6.1f}%  {agg['ML_weighted']:>6.1f}%  "
                         f"{agg['total_FP']:>8}  {agg['total_FN']:>8}  {agg['total_IDSW']:>6}")
       
        # Mean across trackers
        lines.append(f"\n  {'MEAN':>12}  {np.mean([agg['MOTA_agg'] for t in TRACKERS if t in results_by_tracker and split_name in results_by_tracker[t]]):>7.2f}%")
       
        lines.append(f"\n\n{'-'*90}")
        lines.append(f"  {split_name.upper()} SET — Per-Scene Details")
        lines.append(f"{'-'*90}")
       
        for tracker in TRACKERS:
            if tracker not in results_by_tracker or split_name not in results_by_tracker[tracker]:
                continue
           
            results = results_by_tracker[tracker][split_name]
            if not results:
                continue
           
            lines.append(f"\n  {tracker.upper()}:")
            lines.append(f"  {'Scene':>6}  {'MOTA':>8}  {'IDF1':>8}  {'MT%':>7}  {'PT%':>7}  {'ML%':>7}  "
                         f"{'FP':>7}  {'FN':>7}  {'IDSW':>6}")
            lines.append(f"  {'-'*80}")
           
            for r in sorted(results, key=lambda x: x['scene']):
                lines.append(f"  {r['scene']:>6}  {r['MOTA']:>7.2f}%  {r['IDF1']:>7.2f}%  "
                             f"{r['MT']:>6.1f}%  {r['PT']:>6.1f}%  {r['ML']:>6.1f}%  "
                             f"{r['FP']:>7}  {r['FN']:>7}  {r['IDSW']:>6}")
   
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
   
    print(f"\n  💾 Summary saved to: {SUMMARY_FILE}")


def main():
    print(f"\n{'='*90}")
    print(f"  MOT METRICS EVALUATION — {MODEL_NAME}")
    print(f"{'='*90}")
    print(f"  Base path: {BASE_PATH}")
    print(f"  Trackers: {', '.join(TRACKERS)}")
    print(f"  IoU threshold: {IOU_THRESHOLD}")
   
    # Store results: results_by_tracker[tracker][split] = list of results
    results_by_tracker = {}
   
    # Evaluate each tracker
    for tracker in TRACKERS:
        print(f"\n\n{'='*90}")
        print(f"  Evaluating {tracker.upper()}...")
        print(f"{'='*90}")
       
        results_by_tracker[tracker] = {"validation": [], "test": []}
       
        tracker_dir = Path(BASE_PATH) / tracker
        if not tracker_dir.exists():
            print(f"  ⚠️  No results found for {tracker.upper()} at {tracker_dir}")
            continue
       
        # Process validation and test scenes
        for split_name, scenes in [("validation", VAL_SCENES), ("test", TEST_SCENES)]:
            split_dir = tracker_dir / split_name
            if not split_dir.exists():
                continue
           
            print(f"\n  {split_name.upper()} SCENES:")
           
            for sid in scenes:
                scene_dir = split_dir / f"scene{sid}"
                result_file = scene_dir / "results.txt"
                gt_file = f"{sid}/gt_scene{sid}.txt"
               
                # Evaluate
                metrics = evaluate_scene(sid, result_file, gt_file)
               
                if metrics:
                    results_by_tracker[tracker][split_name].append(metrics)
                    print(f"    Scene {sid:2d}: MOTA={metrics['MOTA']:6.2f}%, IDF1={metrics['IDF1']:6.2f}%, "
                          f"MT={metrics['MT']:5.1f}%, PT={metrics['PT']:5.1f}%")
                else:
                    print(f"    Scene {sid:2d}: No results found")
       
        # Print aggregate for this tracker
        for split_name in ["validation", "test"]:
            results = results_by_tracker[tracker][split_name]
            if results:
                agg = compute_aggregate_metrics(results)
                print(f"\n  {split_name.upper()} — {tracker.upper()}:")
                print(f"    Aggregated MOTA: {agg['MOTA_agg']:.2f}%")
                print(f"    Mean IDF1: {agg['IDF1_mean']:.2f}%")
                print(f"    Weighted MT: {agg['MT_weighted']:.1f}%, PT: {agg['PT_weighted']:.1f}%, ML: {agg['ML_weighted']:.1f}%")
                print(f"    Total: FP={agg['total_FP']}, FN={agg['total_FN']}, IDSW={agg['total_IDSW']}")
   
    # Print formatted tables
    print_table(results_by_tracker, "validation")
    print_table(results_by_tracker, "test")
    print_summary_table(results_by_tracker)
   
    # Save results
    save_results_to_csv(results_by_tracker)
    save_summary_to_file(results_by_tracker)
   
    print(f"\n{'='*90}")
    print(f"  ✅ Evaluation complete!")
    print(f"  📊 Summary: {SUMMARY_FILE}")
    print(f"  📊 CSV: {CSV_FILE}")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
