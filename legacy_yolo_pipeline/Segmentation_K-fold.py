from ultralytics import YOLO
from sklearn.model_selection import KFold
import shutil
import yaml
import os
import numpy as np

# Define paths
data_yaml = 'shayan-data-allhemorrhage.yaml'
k = 5  # Number of folds

# Load dataset paths from YAML
with open(data_yaml, 'r') as f:
    data_config = yaml.safe_load(f)

base_path = data_config['path']
image_path = os.path.join(base_path, data_config['train'])
labels_path = os.path.join(base_path, data_config['labels'])

# Verify paths exist
print(f"Verifying paths:")
print(f"Base path exists: {os.path.exists(base_path)}")
print(f"Image path exists: {os.path.exists(image_path)}")
print(f"Labels path exists: {os.path.exists(labels_path)}")

image_list = sorted(os.listdir(image_path))
num_samples = len(image_list)
print(f"\nFound {num_samples} images in dataset")

# KFold setup
kf = KFold(n_splits=k, shuffle=False)

for fold, (train_idx, val_idx) in enumerate(kf.split(image_list)):
    print(f"\n\n{'='*50}")
    print(f"Starting fold {fold+1}/{k}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    
    # Create fold directories
    fold_dir = os.path.abspath(f'temp_fold/fold_{fold}')
    fold_train_images = os.path.join(fold_dir, 'train', 'images')
    fold_train_labels = os.path.join(fold_dir, 'train', 'labels')
    fold_val_images = os.path.join(fold_dir, 'val', 'images')
    fold_val_labels = os.path.join(fold_dir, 'val', 'labels')

    for d in [fold_train_images, fold_train_labels, fold_val_images, fold_val_labels]:
        os.makedirs(d, exist_ok=True)
        print(f"Created directory: {d}")

    # Helper function to copy files with verification
    def copy_files(indices, img_dest, lbl_dest):
        copied = 0
        for idx in indices:
            img_name = image_list[idx]
            base_name = os.path.splitext(img_name)[0]
            
            # Copy image
            src_img = os.path.join(image_path, img_name)
            dst_img = os.path.join(img_dest, img_name)
            shutil.copy(src_img, dst_img)
            
            # Copy label
            lbl_file = f"{base_name}.txt"
            src_lbl = os.path.join(labels_path, lbl_file)
            dst_lbl = os.path.join(lbl_dest, lbl_file)
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, dst_lbl)
                copied += 1
        print(f"Copied {len(indices)} images and {copied} labels to {img_dest}")

    # Copy training and validation data
    print("\nCopying training data...")
    copy_files(train_idx, fold_train_images, fold_train_labels)
    print("\nCopying validation data...")
    copy_files(val_idx, fold_val_images, fold_val_labels)

    # Create fold-specific YAML
    fold_data_config = {
        'path': os.path.abspath(fold_dir),  # Use absolute path to fold directory
        'train': 'train/images',
        'val': 'val/images',
        'nc': data_config['nc'],
        'names': data_config['names']
    }

    fold_yaml_path = os.path.join(fold_dir, 'fold_data.yaml')
    with open(fold_yaml_path, 'w') as f:
        yaml.dump(fold_data_config, f)
    
    print("\nFold YAML configuration:")
    print(yaml.dump(fold_data_config))
    
    # Initialize model with more feedback
    print("\nInitializing model...")
    try:
        model = YOLO('yolo11x-seg.pt')  # Try with official model first
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        continue
    
    # Train with more conservative parameters
    print("\nStarting training...")
    try:
        results = model.train(
            data=fold_yaml_path,
            epochs=100,  # Try with fewer epochs first
            imgsz=512,
            batch=8,  # Reduced batch size
            device='0,1,2,3',  # Try with single GPU first
            verbose=True  # Ensure verbose output
        )
        print(f"\nFold {fold+1} training completed")
    except Exception as e:
        print(f"\nError during training fold {fold+1}: {e}")
    
    # Cleanup
    shutil.rmtree(fold_dir)
    print(f"Cleaned up fold directory: {fold_dir}")

print("\nCross-validation training completed.")
