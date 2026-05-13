# Age & Gender Detection System

A real-time Computer Vision system that detects faces and predicts Age and Gender using OpenCV DNN.

## Features
- **Face Detection**: Uses ResNet SSD (Single Shot Detector) for fast and accurate face tracking.
- **Age Prediction**: Classifies ag e into 8 buckets (0-2, 4-6, ..., 60+).
- **Gender Prediction**: Classifies as Male or Female.

## Project Structure
```
Age & Gender Detection/
│
├── models/               # Contains .caffemodel and .prototxt files
│   └── download_models.py # Script to fetch models
├── detector.py           # Face detection logic
├── predictor.py          # Age/Gender prediction logic
├── main.py               # Main application loop
└── requirements.txt
```

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Models** (One-time setup):
    ```bash
    python models/download_models.py
    ```
    *This downloads the required Caffe models (~50MB) to the `models/` directory.*

## Running the Application

1.  **Start the Webcam Stream**:
    ```bash
    python main.py
    ```
2.  **Stop**: Press `q` to exit.

## Troubleshooting
- **Models not found**: Ensure you ran `models/download_models.py` and that `models/` folder contains 6 files.
