import os
from ultralytics import YOLO

def process_dataset(base_data_path="data", model_path='yolov8n.pt'):
    """
    Automated pipeline to ensure data integrity and generate missing labels.
    """
    model = YOLO(model_path)
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        img_dir = os.path.join(base_data_path, split, 'images')
        lbl_dir = os.path.join(base_data_path, split, 'labels')
        
        os.makedirs(lbl_dir, exist_ok=True)
        
        if not os.path.exists(img_dir):
            print(f"Skipping {split}: Directory not found.")
            continue
            
        images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        print(f"\nProcessing {split} split ({len(images)} images)...")
        
        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            label_file = img_name.rsplit('.', 1)[0] + ".txt"
            label_path = os.path.join(lbl_dir, label_file)
            
            # Check if label exists and is valid
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                continue # Skip, file is already annotated
                
            # Perform Auto-Labeling
            results = model.predict(img_path, conf=0.5, verbose=False)
            
            with open(label_path, 'w') as f:
                for box in results[0].boxes:
                    # Save in YOLO format: class_id x_center y_center width height
                    coords = box.xywhn[0].tolist()
                    class_id = int(box.cls[0])
                    f.write(f"{class_id} {' '.join(map(str, coords))}\n")
        
        print(f"Completed {split} pipeline.")

if __name__ == "__main__":
    # Point directly to your data folder
    process_dataset(base_data_path="D:/Parth/vanco-solution-architecture/asl_detection/data")