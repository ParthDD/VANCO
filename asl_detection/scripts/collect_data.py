import cv2
import os
import string

def capture_frames(class_name, num_images=20):
    # Using your directory structure: ../data/raw/Sign_X
    save_path = f"../data/raw/{class_name}" 
    os.makedirs(save_path, exist_ok=True)
    
    print(f"--- Get ready for: {class_name} ---")
    cap = cv2.VideoCapture(0)
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        cv2.imshow(f"Capturing {class_name} ({count+1}/{num_images})", frame)
        
        # Press Space to capture
        if cv2.waitKey(1) == ord(' '):
            #This creates names like Sign_A_0000.jpg, Sign_A_0001.jpg
            filename = f"{class_name}_{count:04d}.jpg"
            cv2.imwrite(os.path.join(save_path, filename), frame)
            
            count += 1
            print(f"Captured {count}/{num_images}")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    for letter in string.ascii_uppercase:
        capture_frames(f"Sign_{letter}")