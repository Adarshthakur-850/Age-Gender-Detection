# Age and Gender Detection System

A Machine Learning/Computer Vision project that detects human faces in real-time and predicts:

- Age group
- Gender
- Face location

This project uses **Python**, **OpenCV**, and Deep Learning models for real-time age and gender classification through webcam input or uploaded images.

---

## Features

✅ Real-time face detection using webcam  
✅ Predicts gender (Male/Female)  
✅ Predicts approximate age range  
✅ Supports image input detection  
✅ Bounding box visualization on detected faces  
✅ Fast and lightweight implementation  

---

## Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- CNN
- Deep Learning Models

---

## Project Structure

```bash
Age-Gender-Detection/
│
├── models/
│   ├── age_model.h5
│   ├── gender_model.h5
│
├── images/
│   ├── sample1.jpg
│
├── app.py
├── detect.py
├── requirements.txt
├── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Adarshthakur-850/Age-Gender-Detection.git
cd Age-Gender-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### For webcam detection

```bash
python app.py
```

### For image detection

```bash
python detect.py
```

---

## Working Process

1. Capture image/video frame  
2. Detect face using OpenCV  
3. Extract facial region  
4. Pass face image to trained model  
5. Predict age  
6. Predict gender  
7. Display final output on screen  

---

## Model Output Example

```bash
Gender: Male
Age: 21-30
```

---

## Applications

- Smart surveillance systems  
- Retail customer analytics  
- Attendance systems  
- Human-computer interaction  
- Demographic analysis  

---

## Future Improvements

- Improve model accuracy  
- Add emotion detection  
- Deploy using Flask/FastAPI  
- Docker deployment  
- Cloud deployment support  

---

## Author

**Adarsh Thakur**

GitHub: https://github.com/Adarshthakur-850

---

## License

This project is open-source and available under the MIT License.
