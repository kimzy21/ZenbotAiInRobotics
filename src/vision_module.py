"""
Vision Module for DOFBOT Smart Desk-Tidying Robot
Handles object detection using YOLOv8 and color classification using OpenCV
Developer: Kimberley Alexya Ramasamy

UPDATED: 8-class dataset (aa_battery, charger_adapter, eraser, glue_stick, 
         highlighter, pen, sharpener, stapler)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict


class VisionModule:
    """Handle object detection and color classification for desk items."""
    
    def __init__(self, model_path: str = '../models/best.pt'):
        """
        Initialize vision module with trained YOLOv8 model.
        
        Args:
            model_path: Path to trained YOLOv8 model weights
        """
        print(f"🔧 Loading YOLOv8 model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Class names from YOUR training (8 classes)
        self.class_names = [
            'aa_battery',        # 0
            'charger_adapter',   # 1
            'eraser',            # 2
            'glue_stick',        # 3
            'highlighter',       # 4
            'pen',               # 5
            'sharpener',         # 6
            'stapler'            # 7
        ]
        
        # ROYGBIV + Neutral colors for desk organization
        self.color_map = {
            'red': (0, 0, 255),
            'orange': (0, 165, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'indigo': (130, 0, 75),
            'violet': (255, 0, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'gray': (128, 128, 128),
            'brown': (42, 42, 165),
        }
        
        print("✅ Vision module initialized successfully")
        print(f"📦 Loaded {len(self.class_names)} object classes:")
        for i, name in enumerate(self.class_names):
            print(f"   {i}: {name}")
    
    def detect_objects(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in frame using YOLOv8.
        
        Args:
            frame: Input BGR image from camera
            conf_threshold: Minimum confidence threshold (0-1)
            
        Returns:
            List of detected objects with bounding boxes and metadata
        """
        # Run YOLOv8 inference
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                # Calculate center point and dimensions
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                # Store detection info
                detection = {
                    'class_id': cls_id,
                    'class_name': self.class_names[cls_id],
                    'confidence': conf,
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'center': (center_x, center_y),
                    'width': width,
                    'height': height
                }
                
                detections.append(detection)
        
        return detections
    
    def get_object_color(self, frame: np.ndarray, detection: Dict) -> str:
        """
        Determine dominant color of detected object using HSV color space.
        
        Args:
            frame: Input BGR image
            detection: Detection dictionary with bbox information
            
        Returns:
            Color name as string (e.g., 'red', 'blue', 'green')
        """
        x1, y1, x2, y2 = detection['bbox']
        
        # Extract region of interest (ROI)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 'unknown'
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define HSV color ranges for each color
        color_ranges = {
            # Hue ranges in HSV (Hue: 0-180, Saturation: 0-255, Value: 0-255)
            'red': [(0, 100, 100), (10, 255, 255)],  # Red wraps around
            'red2': [(170, 100, 100), (180, 255, 255)],  # Red upper range
            'orange': [(10, 100, 100), (25, 255, 255)],
            'yellow': [(25, 100, 100), (35, 255, 255)],
            'green': [(40, 50, 50), (90, 255, 255)],
            'blue': [(100, 50, 50), (130, 255, 255)],
            'indigo': [(130, 50, 50), (150, 255, 255)],
            'violet': [(150, 50, 50), (170, 255, 255)],
            'white': [(0, 0, 200), (180, 25, 255)],
            'black': [(0, 0, 0), (180, 255, 50)],
            'gray': [(0, 0, 50), (180, 50, 200)],
            'brown': [(10, 100, 20), (20, 255, 200)],
        }
        
        max_pixels = 0
        detected_color = 'unknown'
        
        # Count pixels matching each color range
        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)
            pixels = cv2.countNonZero(mask)
            
            if pixels > max_pixels:
                max_pixels = pixels
                # Handle red's dual range
                if color == 'red2':
                    detected_color = 'red'
                else:
                    detected_color = color
        
        return detected_color
    
    def classify_by_color(self, frame: np.ndarray, detections: List[Dict]) -> Dict[str, List]:
        """
        Organize detected objects into color-based bins.
        
        Args:
            frame: Input BGR image
            detections: List of object detections
            
        Returns:
            Dictionary mapping colors to lists of objects
        """
        color_bins = {}
        
        for detection in detections:
            # Get dominant color for this object
            color = self.get_object_color(frame, detection)
            
            # Add color field to detection
            detection['color'] = color
            
            # Group by color
            if color not in color_bins:
                color_bins[color] = []
            
            color_bins[color].append(detection)
        
        return color_bins
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       show_colors: bool = False) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input BGR image
            detections: List of detections to draw
            show_colors: Whether to show detected colors in labels
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            # Color for bounding box (green)
            color = (0, 255, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(annotated, (center_x, center_y), 5, color, -1)
            
            # Prepare label text
            if show_colors and 'color' in detection:
                label = f"{class_name} ({confidence:.2f}) - {detection['color']}"
            else:
                label = f"{class_name} ({confidence:.2f})"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), 
                         (x1 + label_w, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated
    
    def get_largest_object(self, detections: List[Dict]) -> Dict:
        """
        Find the largest detected object by bounding box area.
        
        Args:
            detections: List of detections
            
        Returns:
            Detection dictionary of largest object, or None if no detections
        """
        if not detections:
            return None
        
        return max(detections, key=lambda d: d['width'] * d['height'])
    
    def filter_by_class(self, detections: List[Dict], class_name: str) -> List[Dict]:
        """
        Filter detections by class name.
        
        Args:
            detections: List of all detections
            class_name: Class name to filter by (e.g., 'pen', 'stapler')
            
        Returns:
            Filtered list of detections
        """
        return [d for d in detections if d['class_name'] == class_name]
    
    def get_color_statistics(self, color_bins: Dict[str, List]) -> Dict:
        """
        Get statistics about detected colors.
        
        Args:
            color_bins: Dictionary of color bins
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_objects': sum(len(objects) for objects in color_bins.values()),
            'colors_detected': len(color_bins),
            'distribution': {color: len(objects) for color, objects in color_bins.items()}
        }
        return stats


def test_vision():
    """Test vision module with webcam."""
    print("🎥 Initializing Vision Module Test...")
    print("Press 'q' to quit, 's' to save current frame")
    print("\n📦 Your 8 object classes:")
    print("   0: aa_battery")
    print("   1: charger_adapter")
    print("   2: eraser")
    print("   3: glue_stick")
    print("   4: highlighter")
    print("   5: pen")
    print("   6: sharpener")
    print("   7: stapler\n")
    
    # Initialize vision module
    vision = VisionModule()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break
        
        # Detect objects
        detections = vision.detect_objects(frame, conf_threshold=0.5)
        
        # Classify by color
        color_bins = vision.classify_by_color(frame, detections)
        
        # Draw detections with colors
        annotated = vision.draw_detections(frame, detections, show_colors=True)
        
        # Add statistics overlay
        y_offset = 30
        cv2.putText(annotated, f"Objects: {len(detections)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        for color, objects in color_bins.items():
            text = f"{color}: {len(objects)}"
            cv2.putText(annotated, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 25
        
        # Display frame
        cv2.imshow('DOFBOT Vision Module Test', annotated)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("👋 Exiting...")
            break
        elif key == ord('s'):
            filename = f'capture_{frame_count:04d}.jpg'
            cv2.imwrite(filename, annotated)
            print(f"💾 Saved: {filename}")
            frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test complete")


if __name__ == "__main__":
    test_vision()