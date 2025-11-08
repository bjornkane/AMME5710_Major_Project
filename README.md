# Real-Time Hand Gesture-Controlled Media Player  
**Team M**  
*Siwon Kang, Varunvarshan Sideshkumar, Arin Adurkar*

---

## Overview

A **touchless media player controller** using **real-time hand gesture recognition** via a standard webcam.  
No extra hardware. No physical contact. Just wave your hand to **play, pause, skip, or adjust volume**.

Built with **classical computer vision** (HSV + YCrCb skin fusion, convex hull, SVM) — achieving **96.67% accuracy** and **<120ms latency** on CPU.

---

## Command line arguments needed to run the code(s):

- python collec_data.py --label label_name --frames xx (eg: python collect_data.py --label play --frames 100)
- python train_data.py
- python benchmarking_resnet_18.py
- python resnet_18_verify.py
- python media_controller.py

---

## Key Features

| Feature | Description |
|-------|-----------|
| **6 Static Gestures** | Play, Pause, Volume Up/Down, Next, Previous |
| **Webcam Only** | No gloves, sensors, or depth cameras |
| **Adaptive Lighting** | Live HSV trackbars + YCrCb fusion |
| **Real-Time** | <120ms end-to-end latency @ 25–30 FPS |
| **Robust Segmentation** | Morphological cleaning + largest contour filtering |
| **Lightweight Classifier** | SVM (RF kernel) — 96.67% accuracy on ~200 samples/class |

---

## Installation

### Requirements
- Python 3.8+
- OpenCV (`cv2`)
- NumPy
- scikit-learn
- PyAutoGUI (for media control)
- imutils

```bash
pip install opencv-python numpy scikit-learn pyautogui imutils
