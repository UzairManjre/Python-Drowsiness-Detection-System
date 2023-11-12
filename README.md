# Python-Drowsiness-Detection-System
This project implements a drowsiness detection system using a pre-trained machine learning model. The system is built using Python, Flask, OpenCV, and TensorFlow. It detects the state of the eyes in real-time from a webcam feed and alerts the user when signs of drowsiness are detected.


# Drowsiness Detection Python Script

## Overview

This Python script implements a drowsiness detection system using a machine learning model built with TensorFlow and OpenCV. The script trains a simple neural network to detect open and closed eyes and provides functionality to use the trained model for real-time drowsiness detection from a webcam feed.

## Features

- Train a drowsiness detection model using TensorFlow and OpenCV.
- Real-time drowsiness detection using the trained model.
- Simple and customizable script for educational and prototyping purposes.

## Requirements

- Python 3.10
- TensorFlow
- OpenCV
- NumPy

## Usage

1. **Training the Model:**

   - Organize your dataset into folders, e.g., `Open_Eyes` and `Closed_Eyes`.
   - Modify the script to specify the path to your dataset and adjust parameters if needed.
   - Run the script to train the model:

     ```bash
     python train_model.py
     ```

2. **Real-time Drowsiness Detection:**

   - After training the model, use the trained model for real-time drowsiness detection:

     ```bash
     python drowsiness_detection.py
     ```

   - The script accesses the webcam, captures frames, and detects the state of the eyes in real-time.

## Customization

- Adjust parameters such as image size, model architecture, and training epochs based on your specific requirements.
- Experiment with different datasets to improve the model's performance.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.

## License

This script is licensed under the [MIT License](LICENSE).

