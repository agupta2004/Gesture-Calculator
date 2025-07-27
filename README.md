
# Gesture-Based Calculator

This is a deep learning-powered calculator that uses hand gestures drawn in the air to input digits and mathematical operations. Using a webcam and MediaPipe, the system captures hand motion, recognizes hand-drawn symbols using a CNN model, and evaluates expressions using simple gestures — like a fist to submit, a palm to evaluate, and a peace sign to clear.

# How It Works

- **Draw in the air** with your index finger — the system captures your motion.
- **Submit a symbol** by making a **fist** — it’s recognized and added to the expression.
- **Show your palm** to **evaluate** the full expression.
- **Peace sign** clears the canvas.

Behind the scenes, it uses:
- OpenCV to access the webcam and track drawing points
- MediaPipe to detect and interpret hand gestures
- A trained CNN model to recognize drawn digits and operators

# Features

- Real-time gesture detection using webcam
- Digit and operator recognition using a trained CNN model
- Custom hand gesture commands to evaluate, clear, and submit
- Interactive drawing canvas and visual feedback

# Tech Stack

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy

# Dataset

- Used **MNIST** (digits 0–9) and custom symbols (`+`, `-`, `*`, `/`) for training the CNN.

# Model Accuracy

The custom model achieved **~98% training accuracy** and **96–97% test accuracy** on the combined dataset of digits and symbols.

# Future Improvements

- Support for multi-digit numbers and parentheses
- Enhanced gesture recognition (e.g., thumbs up to undo)
- Deploy as a mobile/web app

# Author

**Apoorv Gupta**  
