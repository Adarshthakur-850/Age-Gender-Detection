import cv2
import numpy as np

class AgeGenderPredictor:
    def __init__(self, age_proto, age_model, gender_proto, gender_model):
        self.age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
        self.gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
        
        self.AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", 
                            "(38-43)", "(48-53)", "(60-100)"]
        self.GENDER_LIST = ["Male", "Female"]
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    def predict(self, face_img):
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                     self.MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender Prediction
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.GENDER_LIST[gender_preds[0].argmax()]
        
        # Age Prediction
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.AGE_BUCKETS[age_preds[0].argmax()]
        
        return gender, age
