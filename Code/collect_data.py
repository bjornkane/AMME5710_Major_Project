import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime

class GestureDataCollector:
    def __init__(self):
        self.data = []
        self.current_label = None
        self.recording = False
        self.cap = cv2.VideoCapture(0)
        
        # HSV range for skin detection (adjust if needed)
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Gesture labels
        self.gestures = {
            '1': 'play',
            '2': 'pause', 
            '3': 'volume_up',
            '4': 'volume_down'
        }
        
    def detect_hand(self, frame):
        """Segment hand and extract features"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Apply skin color threshold
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Morphological operations to clean up
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5,5), 100)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, mask
            
        # Get largest contour (assume it's the hand)
        hand_contour = max(contours, key=cv2.contourArea)
        
        # Only process if contour is large enough
        if cv2.contourArea(hand_contour) < 3000:
            return None, mask
            
        return hand_contour, mask
    
    def extract_features(self, contour):
        """Extract geometric features from hand contour"""
        # Contour area
        area = cv2.contourArea(contour)
        
        # Perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Convex hull
        hull = cv2.convexHull(contour, returnPoints=False)
        
        # Convexity defects (gaps between fingers)
        defects = cv2.convexityDefects(contour, hull)
        defect_count = 0
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                # Only count significant defects
                if d > 10000:
                    defect_count += 1
        
        # Approximate finger count (defects + 1)
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
        
        # Solidity (area / convex hull area)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area != 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'defect_count': defect_count,
            'finger_count': finger_count,
            'centroid_x': cx,
            'centroid_y': cy,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity
        }
    
    def draw_info(self, frame, features, contour):
        """Draw visualization on frame"""
        if contour is not None:
            # Draw contour
            cv2.drawContours(frame, [contour], 0, (0, 255, 0), 2)
            
            # Draw centroid
            if features:
                cv2.circle(frame, (features['centroid_x'], features['centroid_y']), 5, (255, 0, 0), -1)
                
                # Draw convex hull
                hull = cv2.convexHull(contour)
                cv2.drawContours(frame, [hull], 0, (0, 0, 255), 2)
        
        return frame
    
    def run(self):
        """Main collection loop"""
        print("=== Gesture Data Collector ===")
        print("Press 1-4 to start recording a gesture:")
        print("  1: Play (open palm)")
        print("  2: Pause (closed fist)")
        print("  3: Volume Up (thumbs up)")
        print("  4: Volume Down (thumbs down)")
        print("Press 's' to stop recording")
        print("Press 'q' to quit and save")
        print("================================\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            display = frame.copy()
            
            # Detect hand and extract features
            contour, mask = self.detect_hand(frame)
            features = None
            
            if contour is not None:
                features = self.extract_features(contour)
                display = self.draw_info(display, features, contour)
                
                # If recording, save features
                if self.recording and features:
                    features['label'] = self.current_label
                    self.data.append(features)
            
            # Display info
            status = "RECORDING" if self.recording else "IDLE"
            color = (0, 0, 255) if self.recording else (255, 255, 255)
            cv2.putText(display, f"Status: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if self.recording:
                cv2.putText(display, f"Gesture: {self.gestures[self.current_label]}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Samples: {len([d for d in self.data if d['label'] == self.current_label])}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show frames
            cv2.imshow('Data Collection', display)
            cv2.imshow('Hand Mask', mask)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.recording = False
                self.current_label = None
                print("Stopped recording\n")
            elif chr(key) in self.gestures and not self.recording:
                self.current_label = chr(key)
                self.recording = True
                print(f"Recording gesture: {self.gestures[self.current_label]}")
        
        # Save data
        self.save_data()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
    
    def save_data(self):
        """Save collected data to CSV"""
        if len(self.data) == 0:
            print("No data collected!")
            return
        
        # Create dataframe
        df = pd.DataFrame(self.data)
        
        # Save to CSV
        filename = f"gesture_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        
        print(f"\n=== Data Collection Summary ===")
        print(f"Total samples: {len(df)}")
        print("\nSamples per gesture:")
        print(df['label'].value_counts())
        print(f"\nData saved to: {filename}")

if __name__ == "__main__":
    collector = GestureDataCollector()
    collector.run()