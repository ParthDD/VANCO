from ultralytics import YOLO

def validate_model():
    model = YOLO(r'D:\Parth\vanco-solution-architecture\asl_detection\models\vanco_asl_model\weights\best.pt')
    
    # Validate on test set
    metrics = model.val(data=r'D:\Parth\vanco-solution-architecture\asl_detection\data.yaml', split='test')
    print(f"mAP@50: {metrics.box.map50}")

if __name__ == "__main__":
    validate_model()