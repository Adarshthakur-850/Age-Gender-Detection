import cv2
import numpy as np

class FaceDetector:
    def __init__(self, proto_path, model_path, confidence_threshold=0.5):
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Filter out detections that are out of bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)
                
                faces.append((startX, startY, endX, endY))
        return faces
