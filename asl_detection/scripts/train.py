from ultralytics import YOLO
import torch

# Check if CUDA is available for PyTorch
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Please install the CUDA-enabled version of PyTorch.")


def train_model():
    # Load the model
    model = YOLO(r'D:\Parth\vanco-solution-architecture\asl_detection\yolov8n.pt')
    
    # Train the model
    # Setting device=cpu forces the use of the CPU
    results = model.train(
        data=r'D:\Parth\vanco-solution-architecture\asl_detection\data.yaml', 
        epochs=50, 
        imgsz=640, 
        batch=16,
        device='cpu',  
        project=r'D:\Parth\vanco-solution-architecture\asl_detection\models',
        name='vanco_asl_model'
    )
    print("Training complete using GPU.")

if __name__ == "__main__":
    train_model()