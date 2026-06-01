import os
import shutil
import random

def split_data(raw_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Splits images and corresponding labels from raw_dir into 
    train/val/test folders within target_dir.
    """
    # 1. Setup required directory structure
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(target_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, split, 'labels'), exist_ok=True)

    # 2. Get all images and shuffle them
    files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(files)

    total = len(files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # 3. Distribute files
    for i, file in enumerate(files):
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'

        # Copy image
        shutil.copy(os.path.join(raw_dir, file), os.path.join(target_dir, split, 'images', file))
        
        # Copy corresponding label if it exists
        label_file = os.path.splitext(file)[0] + '.txt'
        label_path = os.path.join(raw_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(target_dir, split, 'labels', label_file))

    print(f"Data split complete: {train_end} train, {val_end-train_end} val, {total-val_end} test.")

if __name__ == "__main__":
    # Update these paths to match your actual project structure
    RAW = r'D:\Parth\vanco-solution-architecture\asl_detection\data\raw'
    TARGET = r'D:\Parth\vanco-solution-architecture\asl_detection\data'
    
    split_data(RAW, TARGET)