import cv2
import numpy as np
import os
from detector import FaceDetector
from predictor import AgeGenderPredictor

def test_pipeline():
    print("Starting verification...")
    
    # Paths (relative to script execution)
    deploy_proto = "models/deploy.prototxt"
    face_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
    age_proto = "models/age_deploy.prototxt"
    age_model = "models/age_net.caffemodel"
    gender_proto = "models/gender_deploy.prototxt"
    gender_model = "models/gender_net.caffemodel"
    
    # 1. Check Files
    missing = []
    for p in [deploy_proto, face_model, age_proto, age_model, gender_proto, gender_model]:
        if not os.path.exists(p):
            missing.append(p)
    
    if missing:
        print(f"FAILED: Missing model files: {missing}")
        return

    print("All model files present.")

    # 2. Load Models
    try:
        detector = FaceDetector(deploy_proto, face_model)
        predictor = AgeGenderPredictor(age_proto, age_model, gender_proto, gender_model)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"FAILED to load models: {e}")
        return

    # 3. Dummy Face Image (227x227 for VGG/Caffe)
    # Create a random image to simulate a face crop
    dummy_face = np.random.randint(0, 255, (227, 227, 3), dtype=np.uint8)
    
    # 4. Predict
    try:
        gender, age = predictor.predict(dummy_face)
        print(f"Prediction Success: Gender={gender}, Age={age}")
    except Exception as e:
        print(f"FAILED prediction: {e}")
        return
        
    print("SUCCESS: Pipeline verified.")

if __name__ == "__main__":
    test_pipeline()
