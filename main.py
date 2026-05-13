import cv2
import time
from detector import FaceDetector
from predictor import AgeGenderPredictor
import os

def main():
    # Paths to models
    # Note: These must match what download_models.py saves
    deploy_proto = "models/deploy.prototxt"
    face_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
    
    age_proto = "models/age_deploy.prototxt"
    age_model = "models/age_net.caffemodel"
    
    gender_proto = "models/gender_deploy.prototxt"
    gender_model = "models/gender_net.caffemodel"
    
    # Check if models exist
    if not all(os.path.exists(p) for p in [deploy_proto, face_model, age_proto, age_model, gender_proto, gender_model]):
        print("Error: Models not found. Please run 'python models/download_models.py' first.")
        return

    print("Loading models...")
    detector = FaceDetector(deploy_proto, face_model)
    predictor = AgeGenderPredictor(age_proto, age_model, gender_proto, gender_model)
    
    cap = cv2.VideoCapture(0)
    print("Starting video stream. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Optional: resize for speed
        # frame = cv2.resize(frame, (640, 480))
        
        # Detect faces
        faces = detector.detect(frame)
        
        for (startX, startY, endX, endY) in faces:
            face_img = frame[startY:endY, startX:endX]
            
            if face_img.size == 0:
                continue
                
            gender, age = predictor.predict(face_img)
            
            label = f"{gender}, {age}"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            
        cv2.imshow("Age & Gender Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
