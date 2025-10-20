import os
import cv2
import csv
import torch
import random
import numpy as np
from ultralytics import YOLO
import motmetrics as mm
from deep_sort_realtime.deepsort_tracker import DeepSort   # ✅ external DeepSORT package


# ==== CONFIG ====
video_path = "37/scene37.mp4"
model_path = "best-11-parasite-1219.pt"
gt_mask_folder = "37/scene37"
output_dir = "eval_results37-deepsort"
tracker_type = "deepsort"   

starting_frame = 499   # first frame to process 
ending_frame = 599     # last frame to process 

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "gt"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "pred"), exist_ok=True)

# Initialize YOLO model
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO(model_path)

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video resolution: {width}x{height}")

# Output video writer
output_video = os.path.join(output_dir, "tracked_evaluation.mp4")
out_vid = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))


def extract_red_boxes(mask_bgr):
    """Extract red bounding boxes from segmentation masks"""
    hsv = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]
    return boxes


def create_proper_gt_file():
    """Create MOT Challenge format ground truth file for a specific frame range"""
    gt_file_path = os.path.join(output_dir, "gt", "gt.txt")

    with open(gt_file_path, "w", newline='') as gt_file:
        gt_writer = csv.writer(gt_file, delimiter=',')

        for frame_count in range(starting_frame, ending_frame + 1):
            mask_path = os.path.join(gt_mask_folder, f"{frame_count}.png")
            if not os.path.exists(mask_path):
                continue

            mask = cv2.imread(mask_path)
            if mask is None:
                continue

            boxes = extract_red_boxes(mask)
            for box_id, (x, y, w, h) in enumerate(boxes, start=1):
                gt_writer.writerow([frame_count - starting_frame + 1, box_id, x, y, w, h, 1, -1, -1, -1])

    print(f"\n✅ Proper MOT format GT file created at {gt_file_path}")
    total_frames = ending_frame - starting_frame + 1
    return gt_file_path, total_frames


def run_tracking_and_save_results(total_frames):
    """Run tracking (BoT-SORT, ByteTrack, or DeepSORT) and save results in MOT format"""
    results_file = os.path.join(output_dir, "pred", "track_results.txt")
    track_colors = {}

    # Start reading video from starting_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

    # Init DeepSORT if selected
    if tracker_type == "deepsort":
        deepsort_tracker = DeepSort(
            max_age=30,
            n_init=2,
            embedder='mobilenet',
            half=True,
            embedder_gpu=torch.cuda.is_available()
        )

    with open(results_file, "w", newline='') as res_file:
        writer = csv.writer(res_file, delimiter=',')

        for idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            frame_id = idx + 1  # Local frame ID for evaluation

            if tracker_type in ["botsort", "bytetrack"]:
                results = model.track(source=frame, tracker=f"{tracker_type}.yaml", persist=True, device=device)
                if results and results[0].boxes is not None and results[0].boxes.id is not None:
                    ids = results[0].boxes.id.int().cpu().tolist()
                    boxes = results[0].boxes.xyxy.cpu().tolist()
                    confs = results[0].boxes.conf.cpu().tolist()

                    for track_id, box, conf in zip(ids, boxes, confs):
                        x1, y1, x2, y2 = map(int, box)
                        w, h = x2 - x1, y2 - y1
                        writer.writerow([frame_id, track_id, x1, y1, w, h, conf, -1, -1, -1])

                        # Visualization
                        if track_id not in track_colors:
                            track_colors[track_id] = (
                                random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255)
                            )
                        color = track_colors[track_id]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)

            elif tracker_type == "deepsort":
                # Run YOLO detection
                results = model(frame, verbose=False)[0]
                detections = []
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

                tracks = deepsort_tracker.update_tracks(detections, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    w, h = x2 - x1, y2 - y1
                    tid = track.track_id
                    conf = 1.0  # DeepSORT doesn't give conf per track
                    writer.writerow([frame_id, tid, x1, y1, w, h, conf, -1, -1, -1])

                    if tid not in track_colors:
                        track_colors[tid] = tuple(random.randint(50, 200) for _ in range(3))
                    color = track_colors[tid]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
                    cv2.putText(frame, f'ID:{tid}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # Draw GT boxes (blue)
            mask_path = os.path.join(gt_mask_folder, f"{starting_frame + idx}.png")
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path)
                if mask is not None:
                    gt_boxes = extract_red_boxes(mask)
                    gt_color = (255, 0, 0)
                    for box_id, (x, y, w, h) in enumerate(gt_boxes, start=1):
                        cv2.rectangle(frame, (x, y), (x + w, y + h), gt_color, 5)
                        cv2.putText(frame, f"GT {box_id}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, gt_color, 4)

            out_vid.write(frame)
            print(f"Processing frame {starting_frame + idx}/{ending_frame}", end='\r')

    print(f"\n✅ Tracking results saved to {results_file}")
    return results_file


def evaluate_tracking(gt_file, results_file):
    """Calculate and display MOT metrics"""
    gt = mm.io.load_motchallenge(gt_file)
    res = mm.io.load_motchallenge(results_file)
    
    acc = mm.utils.compare_to_groundtruth(gt, res, 'iou', distth=0.5)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='Overall')
    
    summary = mm.io.render_summary(
        summary, 
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    print("\n📊 Tracking Evaluation Metrics:")
    print(summary)
    
    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(summary)
    print(f"\n✅ Metrics saved to {metrics_file}")


if __name__ == "__main__":
    gt_file, total_frames = create_proper_gt_file()
    results_file = run_tracking_and_save_results(total_frames)
    evaluate_tracking(gt_file, results_file)
    cap.release()
    out_vid.release()
    print("\n✅ Evaluation completed successfully!")

