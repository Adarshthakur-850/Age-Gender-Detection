import cv2
import time
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from detector import FaceDetector
from predictor import AgeGenderPredictor

def test_runtime():
    print("Initializing Runtime Test...")

    # Paths
    deploy_proto = "models/deploy.prototxt"
    face_model = "models/res10_300x300_ssd_iter_140000.caffemodel"
    age_proto = "models/age_deploy.prototxt"
    age_model = "models/age_net.caffemodel"
    gender_proto = "models/gender_deploy.prototxt"
    gender_model = "models/gender_net.caffemodel"
    
    # Check Models
    if not all(os.path.exists(p) for p in [deploy_proto, face_model, age_proto, age_model, gender_proto, gender_model]):
        print("FAILED: Models missing.")
        return

    # Load Models
    try:
        detector = FaceDetector(deploy_proto, face_model)
        predictor = AgeGenderPredictor(age_proto, age_model, gender_proto, gender_model)
        print("Models loaded.")
    except Exception as e:
        print(f"FAILED to load models: {e}")
        return

    # Mock Video Source (Synthetic frames to avoid camera dependency in test)
    # We create a list of frames, some blank, some with a fake "face"
    print("Starting synthetic video loop...")
    
    for i in range(10):
        # Create a frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw a fake face (just random noise in a rect)
        # The face detector likely won't detect this as a face, 
        # but we testing the loop mechanics and object instantiation.
        # To test actual prediction, we can force a detection or just test the predictor separately (which we did).
        # Let's try to pass a frame that TRIGGERs the predictor by manually cropping a region
        # mimicking a detected face if detector finds nothing.
        
        # Run detection
        try:
            faces = detector.detect(frame)
            print(f"Frame {i}: Detected {len(faces)} faces.")
            
            # If no faces (expected on blank image), let's manually feed a crop to predictor
            # to ensure predictor doesn't crash on repeated calls
            if len(faces) == 0:
                fake_face_roi = np.random.randint(0, 255, (227, 227, 3), dtype=np.uint8)
                gender, age = predictor.predict(fake_face_roi)
        
        except Exception as e:
            print(f"FAILED during loop: {e}")
            return
            
    print("SUCCESS: Runtime test completed without errors.")

if __name__ == "__main__":
    test_runtime()
