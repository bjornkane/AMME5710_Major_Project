# import cv2
# import numpy as np
# import joblib
# from collections import deque
# import pyautogui
# import time
# import subprocess

# class GestureRecognitionSystem:
#     def __init__(self, model_path, scaler_path, mapping_path):
#         """Initialize the real-time gesture recognition system"""
        
#         # Load trained model, scaler, and gesture mapping
#         print("Loading model and components...")
#         self.model = joblib.load(model_path)
#         self.scaler = joblib.load(scaler_path)
#         self.gesture_mapping = joblib.load(mapping_path)
#         print("✓ Model loaded successfully\n")
        
#         # Reverse mapping for predictions (1,2,3,4 -> gesture names)
#         self.label_to_gesture = self.gesture_mapping
        
#         # HSV range for skin detection (same as data collection)
#         self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#         self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
#         # Temporal smoothing - store recent predictions
#         self.prediction_window = deque(maxlen=15)  # Last 15 frames (~0.5 sec at 30fps)
#         self.last_gesture = None
#         self.last_action_time = 0
#         self.action_cooldown = 1.0  # 1 second between actions
        
#         # Statistics
#         self.frame_count = 0
#         self.gesture_counts = {gesture: 0 for gesture in self.gesture_mapping.values()}
        
#     def detect_hand(self, frame):
#         """Segment hand from frame - same as data collection"""
#         # Convert to HSV
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
#         # Apply skin color threshold
#         mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
#         # Morphological operations
#         kernel = np.ones((3,3), np.uint8)
#         mask = cv2.erode(mask, kernel, iterations=2)
#         mask = cv2.dilate(mask, kernel, iterations=2)
#         mask = cv2.GaussianBlur(mask, (5,5), 100)
        
#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#         if len(contours) == 0:
#             return None, mask
            
#         # Get largest contour
#         hand_contour = max(contours, key=cv2.contourArea)
        
#         # Only process if large enough
#         if cv2.contourArea(hand_contour) < 3000:
#             return None, mask
            
#         return hand_contour, mask
    
#     def extract_features(self, contour):
#         """Extract same features as training data"""
#         # Contour area
#         area = cv2.contourArea(contour)
        
#         # Perimeter
#         perimeter = cv2.arcLength(contour, True)
        
#         # Convex hull and defects
#         hull = cv2.convexHull(contour, returnPoints=False)
#         defects = cv2.convexityDefects(contour, hull)
#         defect_count = 0
        
#         if defects is not None:
#             for i in range(defects.shape[0]):
#                 s, e, f, d = defects[i, 0]
#                 if d > 10000:
#                     defect_count += 1
        
#         finger_count = defect_count + 1 if defect_count > 0 else 0
        
#         # Moments for centroid
#         M = cv2.moments(contour)
#         if M['m00'] != 0:
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#         else:
#             cx, cy = 0, 0
        
#         # Bounding rectangle
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = float(w) / h if h != 0 else 0
        
#         # Solidity
#         hull_area = cv2.contourArea(cv2.convexHull(contour))
#         solidity = area / hull_area if hull_area != 0 else 0
        
#         # Return as list in same order as training
#         return [area, perimeter, defect_count, finger_count, 
#                 cx, cy, aspect_ratio, solidity]
    
#     def predict_gesture(self, features):
#         """Predict gesture from features"""
#         # Scale features
#         features_scaled = self.scaler.transform([features])
        
#         # Predict
#         prediction = self.model.predict(features_scaled)[0]
        
#         # Get confidence (if model supports predict_proba)
#         confidence = 0.0
#         if hasattr(self.model, 'predict_proba'):
#             proba = self.model.predict_proba(features_scaled)[0]
#             confidence = max(proba)
        
#         return prediction, confidence
    
#     def smooth_predictions(self, prediction):
#         """Apply temporal smoothing to reduce jitter"""
#         self.prediction_window.append(prediction)
        
#         # Return most common prediction in window
#         if len(self.prediction_window) >= 10:
#             # Count occurrences
#             counts = {}
#             for pred in self.prediction_window:
#                 counts[pred] = counts.get(pred, 0) + 1
            
#             # Return most frequent
#             smoothed = max(counts, key=counts.get)
#             return smoothed
        
#         return prediction
    
#     # def execute_media_control(self, gesture_label):
#     #     """Execute media control action based on gesture"""
#     #     current_time = time.time()
        
#     #     # Check cooldown
#     #     if current_time - self.last_action_time < self.action_cooldown:
#     #         return False
        
#     #     # Get gesture name
#     #     gesture_name = self.label_to_gesture.get(str(gesture_label), "Unknown")
        
#     #     # Only execute if gesture changed
#     #     if gesture_label == self.last_gesture:
#     #         return False
        
#     #     # Execute action based on gesture
#     #     action_executed = False
        
#     #     # if gesture_label == '1':  # Play
#     #     #     pyautogui.press('playpause')
#     #     #     print(f"▶️  Action: PLAY")
#     #     #     action_executed = True
            
#     #     # elif gesture_label == '2':  # Pause
#     #     #     pyautogui.press('playpause')
#     #     #     print(f"⏸️  Action: PAUSE")
#     #     #     action_executed = True
            
#     #     # elif gesture_label == '3':  # Volume Up
#     #     #     pyautogui.press('volumeup')
#     #     #     print(f"🔊 Action: VOLUME UP")
#     #     #     action_executed = True
            
#     #     # elif gesture_label == '4':  # Volume Down
#     #     #     pyautogui.press('volumedown')
#     #     #     print(f"🔉 Action: VOLUME DOWN")
#     #     #     action_executed = True
#     #     if gesture_label == '1':  # Play
#     #         pyautogui.press('space')  # Spacebar works better on Mac
#     #         print(f"▶️  Action: PLAY")
#     #         action_executed = True
    
#     #     elif gesture_label == '2':  # Pause
#     #         pyautogui.press('space')
#     #         print(f"⏸️  Action: PAUSE")
#     #         action_executed = True
    
#     #     elif gesture_label == '3':  # Volume Up
#     #         pyautogui.hotkey('command', 'up')  # Mac volume control
#     #         print(f"🔊 Action: VOLUME UP")
#     #         action_executed = True
    
#     #     elif gesture_label == '4':  # Volume Down
#     #         pyautogui.hotkey('command', 'down')  # Mac volume control
#     #         print(f"🔉 Action: VOLUME DOWN")
#     #         action_executed = True
        
#     #     if action_executed:
#     #         self.last_gesture = gesture_label
#     #         self.last_action_time = current_time
#     #         self.gesture_counts[gesture_name] += 1
            
#     #     return action_executed
#     def execute_media_control(self, gesture_label):
#         """Execute media control action based on gesture"""
#         current_time = time.time()
    
#         # Check cooldown only
#         if current_time - self.last_action_time < self.action_cooldown:
#             return False
    
#         gesture_name = self.label_to_gesture.get(str(gesture_label), "Unknown")
    
#         # REMOVED: gesture change check - execute every time after cooldown
    
#         action_executed = False
    
#         if gesture_label == '1' or gesture_label == '2':
#             subprocess.run(['osascript', '-e', '''
#                 tell application "Google Chrome 2"
#                     activate
#                     tell application "System Events"
#                         keystroke "k"
#                     end tell
#                 end tell
#             '''])
#             print(f"⏯️  Action: PLAY/PAUSE (gesture: {gesture_name})")
#             action_executed = True
        
#         elif gesture_label == '3':
#             subprocess.run(['osascript', '-e', '''
#                 tell application "Google Chrome 2"
#                     activate
#                     tell application "System Events"
#                         key code 126
#                     end tell
#                 end tell
#             '''])
#             print(f"🔊 Action: VOLUME UP")
#             action_executed = True
        
#         elif gesture_label == '4':
#             subprocess.run(['osascript', '-e', '''
#                 tell application "Google Chrome 2"
#                     activate
#                     tell application "System Events"
#                         key code 125
#                     end tell
#                 end tell
#             '''])
#             print(f"🔉 Action: VOLUME DOWN")
#             action_executed = True
    
#         if action_executed:
#             self.last_gesture = gesture_label
#             self.last_action_time = current_time
#             self.gesture_counts[gesture_name] += 1
        
#         return action_executed

    
#     def draw_ui(self, frame, gesture_label, confidence, contour):
#         """Draw UI elements on frame"""
#         h, w = frame.shape[:2]
        
#         # Draw hand contour
#         if contour is not None:
#             cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
#             hull = cv2.convexHull(contour)
#             cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
        
#         # Semi-transparent overlay for info panel
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
#         cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
#         # Current gesture
#         gesture_name = self.label_to_gesture.get(str(gesture_label), "No Hand Detected")
#         cv2.putText(frame, f"Gesture: {gesture_name}", (20, 40),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         # Confidence
#         if confidence > 0:
#             cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 70),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # Last action
#         if self.last_gesture:
#             last_action = self.label_to_gesture.get(str(self.last_gesture), "None")
#             cv2.putText(frame, f"Last Action: {last_action}", (20, 100),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
#         # Frame count
#         cv2.putText(frame, f"Frame: {self.frame_count}", (20, 130),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         # Instructions
#         cv2.putText(frame, "Press 'q' to quit | 's' for stats", (20, 160),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         return frame
    
#     def print_statistics(self):
#         """Print usage statistics"""
#         print("\n" + "="*50)
#         print("SESSION STATISTICS")
#         print("="*50)
#         print(f"Total frames processed: {self.frame_count}")
#         print("\nGesture executions:")
#         for gesture, count in self.gesture_counts.items():
#             print(f"  {gesture}: {count}")
#         print("="*50 + "\n")
    
#     def run(self):
#         """Main recognition loop"""
#         print("\n" + "="*50)
#         print("REAL-TIME GESTURE RECOGNITION SYSTEM")
#         print("="*50)
#         print("\nGesture Controls:")
#         print("  1️⃣  Open Palm      → Play/Pause")
#         print("  2️⃣  Closed Fist    → Play/Pause")
#         print("  3️⃣  Thumbs Up      → Volume Up")
#         print("  4️⃣  Thumbs Down    → Volume Down")
#         print("\nInstructions:")
#         print("  • Show gesture to webcam")
#         print("  • System has 1-second cooldown between actions")
#         print("  • Press 'q' to quit")
#         print("  • Press 's' to show statistics")
#         print("="*50 + "\n")
        
#         cap = cv2.VideoCapture(0)
        
#         if not cap.isOpened():
#             print("❌ Error: Cannot open webcam!")
#             return
        
#         print("✓ Webcam opened successfully")
#         print("🎥 Starting recognition...\n")
        
#         try:
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 self.frame_count += 1
                
#                 # Flip for mirror effect
#                 frame = cv2.flip(frame, 1)
#                 display = frame.copy()
                
#                 # Detect hand
#                 contour, mask = self.detect_hand(frame)
                
#                 gesture_label = None
#                 confidence = 0.0
                
#                 if contour is not None:
#                     # Extract features
#                     features = self.extract_features(contour)
                    
#                     # Predict gesture
#                     raw_prediction, confidence = self.predict_gesture(features)
                    
#                     # Smooth prediction
#                     gesture_label = self.smooth_predictions(raw_prediction)
                    
#                     # Execute media control
#                     self.execute_media_control(gesture_label)
                
#                 # Draw UI
#                 display = self.draw_ui(display, gesture_label, confidence, contour)
                
#                 # Show frames
#                 cv2.imshow('Gesture Recognition', display)
#                 cv2.imshow('Hand Mask', mask)
                
#                 # Handle keyboard
#                 key = cv2.waitKey(1) & 0xFF
                
#                 if key == ord('q'):
#                     print("\nShutting down...")
#                     break
#                 elif key == ord('s'):
#                     self.print_statistics()
        
#         finally:
#             # Cleanup
#             self.print_statistics()
#             cap.release()
#             cv2.destroyAllWindows()
#             print("✓ System shutdown complete")

# if __name__ == "__main__":
#     # Paths to saved model files
#     MODEL_PATH = "gesture_model_svm.pkl"
#     SCALER_PATH = "gesture_scaler.pkl"
#     MAPPING_PATH = "gesture_mapping.pkl"
    
#     # Check if files exist
#     import os
#     if not all(os.path.exists(f) for f in [MODEL_PATH, SCALER_PATH, MAPPING_PATH]):
#         print("❌ Error: Model files not found!")
#         print("Make sure you have:")
#         print("  - gesture_model_svm.pkl")
#         print("  - gesture_scaler.pkl")
#         print("  - gesture_mapping.pkl")
#         print("\nRun train_classifier.py first!")
#     else:
#         system = GestureRecognitionSystem(MODEL_PATH, SCALER_PATH, MAPPING_PATH)
#         system.run()












# import cv2
# import numpy as np
# import joblib
# from collections import deque
# import pyautogui
# import time

# class GestureRecognitionSystem:
#     def __init__(self, model_path, scaler_path, mapping_path):
#         """Initialize the real-time gesture recognition system"""
        
#         # Load trained model, scaler, and gesture mapping
#         print("Loading model and components...")
#         self.model = joblib.load(model_path)
#         self.scaler = joblib.load(scaler_path)
#         self.gesture_mapping = joblib.load(mapping_path)
#         print("✓ Model loaded successfully\n")
        
#         # Reverse mapping for predictions (1,2,3,4 -> gesture names)
#         self.label_to_gesture = self.gesture_mapping
        
#         # HSV range for skin detection (same as data collection)
#         self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#         self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
#         # Temporal smoothing - store recent predictions
#         self.prediction_window = deque(maxlen=15)  # Last 15 frames (~0.5 sec at 30fps)
#         self.last_gesture = None
#         self.last_action_time = 0
#         self.action_cooldown = 1.0  # 1 second between actions
        
#         # Statistics
#         self.frame_count = 0
#         self.gesture_counts = {gesture: 0 for gesture in self.gesture_mapping.values()}
        
#     def detect_hand(self, frame):
#         """Segment hand from frame - same as data collection"""
#         # Convert to HSV
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
#         # Apply skin color threshold
#         mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
#         # Morphological operations
#         kernel = np.ones((3,3), np.uint8)
#         mask = cv2.erode(mask, kernel, iterations=2)
#         mask = cv2.dilate(mask, kernel, iterations=2)
#         mask = cv2.GaussianBlur(mask, (5,5), 100)
        
#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#         if len(contours) == 0:
#             return None, mask
            
#         # Get largest contour
#         hand_contour = max(contours, key=cv2.contourArea)
        
#         # Only process if large enough
#         if cv2.contourArea(hand_contour) < 3000:
#             return None, mask
            
#         return hand_contour, mask
    
#     def extract_features(self, contour):
#         """Extract same features as training data"""
#         # Contour area
#         area = cv2.contourArea(contour)
        
#         # Perimeter
#         perimeter = cv2.arcLength(contour, True)
        
#         # Convex hull and defects
#         hull = cv2.convexHull(contour, returnPoints=False)
#         defects = cv2.convexityDefects(contour, hull)
#         defect_count = 0
        
#         if defects is not None:
#             for i in range(defects.shape[0]):
#                 s, e, f, d = defects[i, 0]
#                 if d > 10000:
#                     defect_count += 1
        
#         finger_count = defect_count + 1 if defect_count > 0 else 0
        
#         # Moments for centroid
#         M = cv2.moments(contour)
#         if M['m00'] != 0:
#             cx = int(M['m10'] / M['m00'])
#             cy = int(M['m01'] / M['m00'])
#         else:
#             cx, cy = 0, 0
        
#         # Bounding rectangle
#         x, y, w, h = cv2.boundingRect(contour)
#         aspect_ratio = float(w) / h if h != 0 else 0
        
#         # Solidity
#         hull_area = cv2.contourArea(cv2.convexHull(contour))
#         solidity = area / hull_area if hull_area != 0 else 0
        
#         # Return as list in same order as training
#         return [area, perimeter, defect_count, finger_count, 
#                 cx, cy, aspect_ratio, solidity]
    
#     def predict_gesture(self, features):
#         """Predict gesture from features"""
#         # Scale features
#         features_scaled = self.scaler.transform([features])
        
#         # Predict
#         prediction = self.model.predict(features_scaled)[0]
        
#         # Get confidence (if model supports predict_proba)
#         confidence = 0.0
#         if hasattr(self.model, 'predict_proba'):
#             proba = self.model.predict_proba(features_scaled)[0]
#             confidence = max(proba)
        
#         return prediction, confidence
    
#     def smooth_predictions(self, prediction):
#         """Apply temporal smoothing to reduce jitter"""
#         self.prediction_window.append(prediction)
        
#         # Return most common prediction in window
#         if len(self.prediction_window) >= 10:
#             # Count occurrences
#             counts = {}
#             for pred in self.prediction_window:
#                 counts[pred] = counts.get(pred, 0) + 1
            
#             # Return most frequent
#             smoothed = max(counts, key=counts.get)
#             return smoothed
        
#         return prediction
    
#     def execute_media_control(self, gesture_label):
#         """Execute media control action based on gesture - Windows Version"""
#         current_time = time.time()
        
#         # Check cooldown
#         if current_time - self.last_action_time < self.action_cooldown:
#             return False
        
#         # Get gesture name
#         gesture_name = self.label_to_gesture.get(str(gesture_label), "Unknown")
        
#         # Only execute if gesture changed
#         if gesture_label == self.last_gesture:
#             return False
        
#         # Execute action based on gesture
#         action_executed = False
        
#         if gesture_label == '1':  # Play
#             pyautogui.press('playpause')
#             print(f"▶️  Action: PLAY")
#             action_executed = True
            
#         elif gesture_label == '2':  # Pause
#             pyautogui.press('playpause')
#             print(f"⏸️  Action: PAUSE")
#             action_executed = True
            
#         elif gesture_label == '3':  # Volume Up
#             pyautogui.press('volumeup')
#             print(f"🔊 Action: VOLUME UP")
#             action_executed = True
            
#         elif gesture_label == '4':  # Volume Down
#             pyautogui.press('volumedown')
#             print(f"🔉 Action: VOLUME DOWN")
#             action_executed = True
        
#         if action_executed:
#             self.last_gesture = gesture_label
#             self.last_action_time = current_time
#             self.gesture_counts[gesture_name] += 1
            
#         return action_executed
    
#     def draw_ui(self, frame, gesture_label, confidence, contour):
#         """Draw UI elements on frame"""
#         h, w = frame.shape[:2]
        
#         # Draw hand contour
#         if contour is not None:
#             cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
#             hull = cv2.convexHull(contour)
#             cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
        
#         # Semi-transparent overlay for info panel
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
#         cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
#         # Current gesture
#         gesture_name = self.label_to_gesture.get(str(gesture_label), "No Hand Detected")
#         cv2.putText(frame, f"Gesture: {gesture_name}", (20, 40),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
#         # Confidence
#         if confidence > 0:
#             cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 70),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
#         # Last action
#         if self.last_gesture:
#             last_action = self.label_to_gesture.get(str(self.last_gesture), "None")
#             cv2.putText(frame, f"Last Action: {last_action}", (20, 100),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
#         # Frame count
#         cv2.putText(frame, f"Frame: {self.frame_count}", (20, 130),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         # Instructions
#         cv2.putText(frame, "Press 'q' to quit | 's' for stats", (20, 160),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
#         return frame
    
#     def print_statistics(self):
#         """Print usage statistics"""
#         print("\n" + "="*50)
#         print("SESSION STATISTICS")
#         print("="*50)
#         print(f"Total frames processed: {self.frame_count}")
#         print("\nGesture executions:")
#         for gesture, count in self.gesture_counts.items():
#             print(f"  {gesture}: {count}")
#         print("="*50 + "\n")
    
#     def run(self):
#         """Main recognition loop"""
#         print("\n" + "="*50)
#         print("REAL-TIME GESTURE RECOGNITION SYSTEM")
#         print("="*50)
#         print("\nGesture Controls:")
#         print("  1️⃣  Open Palm      → Play/Pause")
#         print("  2️⃣  Closed Fist    → Play/Pause")
#         print("  3️⃣  Thumbs Up      → Volume Up")
#         print("  4️⃣  Thumbs Down    → Volume Down")
#         print("\nInstructions:")
#         print("  • Show gesture to webcam")
#         print("  • System has 1-second cooldown between actions")
#         print("  • Press 'q' to quit")
#         print("  • Press 's' to show statistics")
#         print("="*50 + "\n")
        
#         cap = cv2.VideoCapture(0)
        
#         if not cap.isOpened():
#             print("❌ Error: Cannot open webcam!")
#             return
        
#         print("✓ Webcam opened successfully")
#         print("🎥 Starting recognition...\n")
        
#         try:
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 self.frame_count += 1
                
#                 # Flip for mirror effect
#                 frame = cv2.flip(frame, 1)
#                 display = frame.copy()
                
#                 # Detect hand
#                 contour, mask = self.detect_hand(frame)
                
#                 gesture_label = None
#                 confidence = 0.0
                
#                 if contour is not None:
#                     # Extract features
#                     features = self.extract_features(contour)
                    
#                     # Predict gesture
#                     raw_prediction, confidence = self.predict_gesture(features)
                    
#                     # Smooth prediction
#                     gesture_label = self.smooth_predictions(raw_prediction)
                    
#                     # Execute media control
#                     self.execute_media_control(gesture_label)
                
#                 # Draw UI
#                 display = self.draw_ui(display, gesture_label, confidence, contour)
                
#                 # Show frames
#                 cv2.imshow('Gesture Recognition', display)
#                 cv2.imshow('Hand Mask', mask)
                
#                 # Handle keyboard
#                 key = cv2.waitKey(1) & 0xFF
                
#                 if key == ord('q'):
#                     print("\nShutting down...")
#                     break
#                 elif key == ord('s'):
#                     self.print_statistics()
        
#         finally:
#             # Cleanup
#             self.print_statistics()
#             cap.release()
#             cv2.destroyAllWindows()
#             print("✓ System shutdown complete")

# if __name__ == "__main__":
#     # Paths to saved model files
#     MODEL_PATH = "gesture_model_svm.pkl"
#     SCALER_PATH = "gesture_scaler.pkl"
#     MAPPING_PATH = "gesture_mapping.pkl"
    
#     # Check if files exist
#     import os
#     if not all(os.path.exists(f) for f in [MODEL_PATH, SCALER_PATH, MAPPING_PATH]):
#         print("❌ Error: Model files not found!")
#         print("Make sure you have:")
#         print("  - gesture_model_svm.pkl")
#         print("  - gesture_scaler.pkl")
#         print("  - gesture_mapping.pkl")
#         print("\nRun train_classifier.py first!")
#     else:
#         system = GestureRecognitionSystem(MODEL_PATH, SCALER_PATH, MAPPING_PATH)
#         system.run()














import cv2
import numpy as np
import joblib
from collections import deque
import pyautogui
import time

class GestureRecognitionSystem:
    def __init__(self, model_path, scaler_path, mapping_path):
        """Initialize the real-time gesture recognition system"""
        
        # Load trained model, scaler, and gesture mapping
        print("Loading model and components...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.gesture_mapping = joblib.load(mapping_path)
        print("✓ Model loaded successfully\n")
        
        # Reverse mapping for predictions (1,2,3,4 -> gesture names)
        self.label_to_gesture = self.gesture_mapping
        
        # HSV range for skin detection (same as data collection)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Temporal smoothing - store recent predictions
        self.prediction_window = deque(maxlen=15)  # Last 15 frames (~0.5 sec at 30fps)
        self.last_gesture = None
        self.last_action_time = 0
        self.action_cooldown = 1.0  # 1 second between actions
        
        # Statistics
        self.frame_count = 0
        self.gesture_counts = {gesture: 0 for gesture in self.gesture_mapping.values()}
        
    def detect_hand(self, frame):
        """Segment hand from frame - same as data collection"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply skin color threshold
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5,5), 100)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, mask
            
        # Get largest contour
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Only process if large enough
        if cv2.contourArea(hand_contour) < 3000:
            return None, mask
            
        return hand_contour, mask
    
    def extract_features(self, contour):
        """Extract same features as training data"""
        # Contour area
        area = cv2.contourArea(contour)
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Convex hull and defects
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        defect_count = 0
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                if d > 10000:
                    defect_count += 1
        
        finger_count = defect_count + 1 if defect_count > 0 else 0
        
        # Moments for centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Solidity
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area != 0 else 0
        
        # Return as list in same order as training
        return [area, perimeter, defect_count, finger_count, 
                cx, cy, aspect_ratio, solidity]
    
    def predict_gesture(self, features):
        """Predict gesture from features"""
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Get confidence (if model supports predict_proba)
        confidence = 0.0
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[0]
            confidence = max(proba)
        
        return prediction, confidence
    
    def smooth_predictions(self, prediction):
        """Apply temporal smoothing to reduce jitter"""
        self.prediction_window.append(prediction)
        
        # Return most common prediction in window
        if len(self.prediction_window) >= 10:
            # Count occurrences
            counts = {}
            for pred in self.prediction_window:
                counts[pred] = counts.get(pred, 0) + 1
            
            # Return most frequent
            smoothed = max(counts, key=counts.get)
            return smoothed
        
        return prediction
    
    def execute_media_control(self, gesture_label):
        """Execute media control action based on gesture - Windows Version"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_action_time < self.action_cooldown:
            return False
        
        # Get gesture name
        gesture_name = self.label_to_gesture.get(str(gesture_label), "Unknown")
        
        # Execute action based on gesture
        action_executed = False
        
        if gesture_label == '1':  # Play
            pyautogui.press('space')
            print(f"▶️  Action: PLAY")
            action_executed = True
            
        elif gesture_label == '2':  # Pause
            pyautogui.press('space')
            print(f"⏸️  Action: PAUSE")
            action_executed = True
            
        elif gesture_label == '3':  # Volume Up
            pyautogui.press('up')
            print(f"🔊 Action: VOLUME UP")
            action_executed = True
            
        elif gesture_label == '4':  # Volume Down
            pyautogui.press('down')
            print(f"🔉 Action: VOLUME DOWN")
            action_executed = True
        
        if action_executed:
            self.last_gesture = gesture_label
            self.last_action_time = current_time
            self.gesture_counts[gesture_name] += 1
            
        return action_executed
    
    def draw_ui(self, frame, gesture_label, confidence, contour):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Draw hand contour
        if contour is not None:
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
            hull = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
        
        # Semi-transparent overlay for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Current gesture
        gesture_name = self.label_to_gesture.get(str(gesture_label), "No Hand Detected")
        cv2.putText(frame, f"Gesture: {gesture_name}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Confidence
        if confidence > 0:
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Last action
        if self.last_gesture:
            last_action = self.label_to_gesture.get(str(self.last_gesture), "None")
            cv2.putText(frame, f"Last Action: {last_action}", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        # Frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 's' for stats", (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def print_statistics(self):
        """Print usage statistics"""
        print("\n" + "="*50)
        print("SESSION STATISTICS")
        print("="*50)
        print(f"Total frames processed: {self.frame_count}")
        print("\nGesture executions:")
        for gesture, count in self.gesture_counts.items():
            print(f"  {gesture}: {count}")
        print("="*50 + "\n")
    
    def run(self):
        """Main recognition loop"""
        print("\n" + "="*50)
        print("REAL-TIME GESTURE RECOGNITION SYSTEM")
        print("="*50)
        print("\nGesture Controls:")
        print("  1️⃣  Open Palm      → Play/Pause")
        print("  2️⃣  Closed Fist    → Play/Pause")
        print("  3️⃣  Thumbs Up      → Volume Up")
        print("  4️⃣  Thumbs Down    → Volume Down")
        print("\nInstructions:")
        print("  • Show gesture to webcam")
        print("  • System has 1-second cooldown between actions")
        print("  • Press 'q' to quit")
        print("  • Press 's' to show statistics")
        print("="*50 + "\n")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot open webcam!")
            return
        
        print("✓ Webcam opened successfully")
        print("🎥 Starting recognition...\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                
                # Flip for mirror effect
                frame = cv2.flip(frame, 1)
                display = frame.copy()
                
                # Detect hand
                contour, mask = self.detect_hand(frame)
                
                gesture_label = None
                confidence = 0.0
                
                if contour is not None:
                    # Extract features
                    features = self.extract_features(contour)
                    
                    # Predict gesture
                    raw_prediction, confidence = self.predict_gesture(features)
                    
                    # Smooth prediction
                    gesture_label = self.smooth_predictions(raw_prediction)
                    print(f"DEBUG: Detected gesture = {gesture_label}, Name = {self.label_to_gesture.get(str(gesture_label), 'Unknown')}")  # ADD THIS

                    
                    # Execute media control
                    self.execute_media_control(gesture_label)
                
                # Draw UI
                display = self.draw_ui(display, gesture_label, confidence, contour)
                
                # Show frames
                cv2.imshow('Gesture Recognition', display)
                cv2.imshow('Hand Mask', mask)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nShutting down...")
                    break
                elif key == ord('s'):
                    self.print_statistics()
        
        finally:
            # Cleanup
            self.print_statistics()
            cap.release()
            cv2.destroyAllWindows()
            print("✓ System shutdown complete")

if __name__ == "__main__":
    # Paths to saved model files
    MODEL_PATH = "gesture_model_svm.pkl"
    SCALER_PATH = "gesture_scaler.pkl"
    MAPPING_PATH = "gesture_mapping.pkl"
    
    # Check if files exist
    import os
    if not all(os.path.exists(f) for f in [MODEL_PATH, SCALER_PATH, MAPPING_PATH]):
        print("❌ Error: Model files not found!")
        print("Make sure you have:")
        print("  - gesture_model_svm.pkl")
        print("  - gesture_scaler.pkl")
        print("  - gesture_mapping.pkl")
        print("\nRun train_classifier.py first!")
    else:
        system = GestureRecognitionSystem(MODEL_PATH, SCALER_PATH, MAPPING_PATH)
        system.run()