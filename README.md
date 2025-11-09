# Real-Time Hand Gesture-Controlled Media Player  
**Team M**  
*Siwon Kang, Varunvarshan Sideshkumar, Arin Adurkar*

---

## Overview

A **touchless media player controller** using **real-time hand gesture recognition** via a standard webcam.  
No extra hardware. No physical contact. Just wave your hand to **play, pause, skip, or adjust volume**.

---

## Source Files and Responsibilities

| File | Description |
|------|--------------|
| **collect_data.py** | Real-time data collector; builds HSV and YCrCb masks, optionally applies MediaPipe ROI gating, fuses masks, extracts geometric and motion features, saves raw/mask/crop/overlay images, and logs CSVs. |
| **train_data.py** | Handles classical training and selection; standardizes numeric features, trains SVM/KNN/DecisionTree/RandomForest classifiers with stratified k-fold validation, and saves trained artifacts: `gesture_scaler.pkl`, `gesture_model.pkl`, `label_map.npy`, and `feature_names.npy`. |
| **media_controller.py** | Runtime gesture controller; reproduces mask fusion pipeline, isolates the main hand blob, detects convexity defects and fingertip trails, computes rotation direction (CW/CCW), and triggers media key events through stable-episode logic. |
| **benchmarking_resnet_18.py** | Deep-learning baseline; prepares datasets with `ImageFolder`, performs deterministic train/val/test splits, fine-tunes a ResNet-18 model, reports metrics, and exports ONNX models for deployment. |
| **resnet_18_verify.py** | Artifact verification; reloads weights and data, recomputes classification metrics, performs sanity checks, and validates PyTorch vs ONNX inference parity. |
 
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
