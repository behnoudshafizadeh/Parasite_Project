#!/usr/bin/env python
"""
Optimized YOLOv8-SwinT Training Script for Parasite Detection
With AMP check completely disabled
"""

from ultralytics import YOLO
import torch
import os
import subprocess
import sys

# Disable ALL checks at the very beginning
os.environ['ULTRALYTICS_DISABLE_AMP_CHECK'] = '1'
os.environ['ULTRALYTICS_DISABLE_VERSION_CHECK'] = '1'

# Also try to disable through torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = False

def check_install_seaborn():
    """Check if seaborn is installed, install if not"""
    try:
        import seaborn
        print("✅ seaborn already installed")
    except ImportError:
        print("📦 Installing seaborn for better visualizations...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn", "matplotlib", "pandas"])
        print("✅ seaborn installed successfully")

def print_system_info():
    """Print system and GPU information"""
    print("="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
    print("="*60)

def create_assets():
    """Create assets directory and dummy image if needed"""
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultralytics', 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    bus_path = os.path.join(assets_dir, 'bus.jpg')
    if not os.path.exists(bus_path):
        print(f"📝 Creating dummy image at: {bus_path}")
        try:
            import urllib.request
            urllib.request.urlretrieve('https://raw.githubusercontent.com/ultralytics/assets/main/bus.jpg', bus_path)
            print("✅ Downloaded bus.jpg")
        except:
            # Create minimal valid JPEG
            with open(bus_path, 'wb') as f:
                f.write(b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\x01\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xd2\xcf \xff\xd9')
            print("✅ Created dummy JPEG")

def train_parasite_model():
    """Main training function with optimized augmentations"""
    
    # First, ensure assets exist
    create_assets()
    
    # Check and install dependencies
    check_install_seaborn()
    
    # Print system info
    print_system_info()
    
    # Dataset path
    dataset_path = '/home/bshafizadeh/nas/shayan/yolo_parasite_segmentation/Yolov8-IST/ultralytics-main/datasets/parasite.yaml'
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please update the dataset_path variable in the script")
        return
    
    print(f"📂 Using dataset: {dataset_path}")
    print("="*60)
    print("STARTING OPTIMIZED TRAINING")
    print("="*60)
    
    # Load the custom segmentation model
    model = YOLO('yolov8-SwinT.yaml')
    
    # PHASE 1: Initial training with frozen backbone
    print("\n🔷 PHASE 1: Training with frozen backbone (50 epochs)")
    print("-" * 40)
    
    phase1_results = model.train(
        # Basic settings
        data=dataset_path,
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        
        # Project naming
        project='parasite_segmentation',
        name='phase1_frozen',
        exist_ok=False,
        
        # Optimizer settings
        optimizer='SGD',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Freeze backbone (first 10 layers)
        freeze=10,
        
        # Learning rate schedule
        cos_lr=True,
        
        # Augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.3,
        auto_augment='randaugment',
        
        # Training optimizations
        amp=False,  # ⚠️ DISABLED AMP to avoid check
        cache=False,
        patience=20,
        save=True,
        save_period=10,
        verbose=True,
    )
    
    print("\n✅ Phase 1 completed!")
    
    # PHASE 2: Fine-tuning entire model
    print("\n🔷 PHASE 2: Fine-tuning entire model (150 epochs)")
    print("-" * 40)
    
    phase2_results = model.train(
        # Basic settings
        data=dataset_path,
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        
        # Project naming
        project='parasite_segmentation',
        name='phase2_finetune',
        
        # Resume from phase 1
        resume=True,
        
        # Optimizer settings
        optimizer='SGD',
        lr0=0.001,
        lrf=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # Don't freeze anything
        freeze=0,
        
        # Learning rate schedule
        cos_lr=True,
        
        # Augmentation settings
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.15,
        scale=0.5,
        shear=3.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.3,
        copy_paste=0.4,
        auto_augment='randaugment',
        
        # Close mosaic in later epochs
        close_mosaic=10,
        
        # Training optimizations
        amp=False,  # ⚠️ Keep AMP disabled
        cache=False,
        patience=30,
        save=True,
        save_period=10,
        verbose=True,
    )
    
    print("\n✅ Phase 2 completed!")
    
    # Final evaluation
    print("\n🔷 Final evaluation on validation set")
    print("-" * 40)
    
    best_model_path = 'parasite_segmentation/phase2_finetune/weights/best.pt'
    if os.path.exists(best_model_path):
        best_model = YOLO(best_model_path)
        
        metrics = best_model.val(
            data=dataset_path,
            device=0,
            split='val',
            conf=0.25,
            iou=0.6,
        )
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Box mAP50: {metrics.box.map50:.4f}")
        print(f"Box mAP50-95: {metrics.box.map:.4f}")
        print(f"Mask mAP50: {metrics.seg.map50:.4f}")
        print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
        print("="*60)

if __name__ == "__main__":
    train_parasite_model()
