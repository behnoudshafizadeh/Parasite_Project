"""
RF-DETR Segmentation - Fine-tuning on Custom Dataset (Multi-GPU)
=================================================================
With memory optimizations to survive validation OOM
"""


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Synchronous execution for better error handling

import gc
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import supervision as sv
from PIL import Image
import torch
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
#  CONFIGURATION - REDUCED FOR MEMORY
# ─────────────────────────────────────────
DATASET_DIR   = "./dataset"
OUTPUT_DIR    = "./output_nano"
EPOCHS        = 100
BATCH_SIZE    = 16                          # Reduced from 24 to 16 (4 per GPU)
GRAD_ACCUM    = 3                           # Increase grad accum to maintain effective batch 48
LEARNING_RATE = 1e-4
THRESHOLD     = 0.5
NUM_VIZ       = 16
CLASS_NAMES   = ["parasite"]
# ─────────────────────────────────────────


def print_gpu_info():
    """Print information about available GPUs."""
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("  GPU Configuration")
        print("=" * 60)
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"  Total GPUs: {torch.cuda.device_count()}")
        print("=" * 60 + "\n")
        return True
    return False


def cleanup_gpu_memory(*objects, verbose=False):
    """Free GPU memory after training."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def annotate(image, detections, classes):
    """Draw bounding boxes + masks + labels on an image."""
    image = image.convert("RGB")
    annotated = sv.MaskAnnotator().annotate(image, detections)
    labels = [f"{classes.get(cid, str(cid))}: {conf:.2f}" 
              for cid, conf in zip(detections.class_id, detections.confidence)]
    annotated = sv.BoxAnnotator().annotate(annotated, detections)
    annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
    return annotated


def train():
    from rfdetr import RFDETRSegNano

    print("=" * 60)
    print("  RF-DETR Segmentation — Multi-GPU Training (Memory Optimized)")
    print("=" * 60)
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  Total Batch Size : {BATCH_SIZE}")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"  Per-GPU Batch    : {BATCH_SIZE // num_gpus}")
    print(f"  Grad Accum       : {GRAD_ACCUM}")
    print(f"  Effective Batch  : {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Classes          : {CLASS_NAMES}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clear memory before training
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Initialize model
    model = RFDETRSegNano(num_classes=len(CLASS_NAMES))
    
    # Train with aggressive memory optimizations
    # Note: We can't fully disable eval, but we can reduce its memory footprint
    model.train(
        dataset_dir=DATASET_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM,
        lr=LEARNING_RATE,
        output_dir=OUTPUT_DIR,
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        num_workers=0,
        pin_memory=False,
        amp=True,                          # Mixed precision saves memory
        eval_max_dets=100,                 # Reduce evaluation detections
        mask_point_sample_ratio=8,         # Reduce mask sample ratio (default 16)
        sync_bn=False,                     # Disable sync BN to save memory
    )

    print("Training complete!")
    return model


def evaluate():
    from rfdetr import RFDETRSegNano

    print("\n" + "=" * 60)
    print("  Starting Evaluation on Test Set")
    print("=" * 60)
    
    # Free memory before evaluation
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Find checkpoint
    checkpoint = os.path.join(OUTPUT_DIR, "checkpoint_best_total.pth")
    if not os.path.exists(checkpoint):
        # Try to find any checkpoint
        pth_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pth')]
        if pth_files:
            checkpoint = os.path.join(OUTPUT_DIR, pth_files[0])
            print(f"Found checkpoint: {checkpoint}")
        else:
            print(f"No checkpoint found in {OUTPUT_DIR}")
            return

    print(f"\nLoading checkpoint: {checkpoint}")
    model = RFDETRSegNano(pretrain_weights=checkpoint)
    model.optimize_for_inference()
    
    # Set to evaluation mode
    model.eval()
    
    # Use smaller batch for inference
    torch.cuda.empty_cache()

    # Load test dataset
    test_ann = os.path.join(DATASET_DIR, "test", "_annotations.coco.json")
    test_img = os.path.join(DATASET_DIR, "test", "images")
    
    if not os.path.exists(test_img):
        test_img = os.path.join(DATASET_DIR, "test")
    
    ds_test = sv.DetectionDataset.from_coco(
        images_directory_path=test_img,
        annotations_path=test_ann,
        force_masks=True,
    )

    class_map = {i: name for i, name in enumerate(ds_test.classes)}
    print(f"Test set: {len(ds_test)} images | Classes: {list(class_map.values())}")

    # Run predictions on random sample - process one by one to save memory
    n = min(NUM_VIZ, len(ds_test))
    annotated_images = []
    indices = random.sample(range(len(ds_test)), n)

    print("\nGenerating predictions (one image at a time)...")
    for idx in indices:
        try:
            path, _, _ = ds_test[idx]
            image = Image.open(path)
            detections = model.predict(image, threshold=THRESHOLD)
            annotated = annotate(image, detections, class_map)
            annotated_images.append(annotated)
            # Clear cache after each image to prevent memory buildup
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Error processing image {idx}: {e}")
            continue

    if not annotated_images:
        print("No images could be processed. Skipping visualization.")
        return

    # Display results in a grid
    n = len(annotated_images)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (ax, img) in enumerate(zip(axes, annotated_images)):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Sample {i+1}", fontsize=10)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("RF-DETR Segmentation — Test Set Predictions", fontsize=14, fontweight='bold')
    plt.tight_layout()

    viz_path = os.path.join(OUTPUT_DIR, "test_predictions.png")
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved to: {viz_path}")
    
    # Print memory summary
    if torch.cuda.is_available():
        print(f"\nGPU Memory Summary:")
        print(f"  Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"  Max memory reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    
    cleanup_gpu_memory(model, verbose=True)


if __name__ == "__main__":
    # Print GPU information
    print_gpu_info()
    
    # Ask for confirmation
    if torch.cuda.device_count() > 1:
        response = input(f"\nFound {torch.cuda.device_count()} GPUs. Start training? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Training cancelled.")
            exit()
    
    # Train
    train()
    
    # Evaluate after training
    evaluate()
    
    print("\n" + "=" * 60)
    print("  ✅ Training and Evaluation Complete!")
    print("=" * 60)
