import cv2
from ultralytics import YOLO

def run_demo():
    # Load your custom trained model
    model = YOLO(r'D:\Parth\vanco-solution-architecture\asl_detection\models\vanco_asl_model\weights\best.pt')
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Check if webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Vanco ASL Detection... Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Run inference with stream=True
        # This returns a generator, so we iterate through it
        results = model(frame, stream=True, conf=0.5)
        
        # Iterate over the results (there is only one result per frame)
        for r in results:
            # Plot results on the current frame
            annotated_frame = r.plot()
            
            # Display the frame
            cv2.imshow("Vanco ASL Detection Utility", annotated_frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_demo()