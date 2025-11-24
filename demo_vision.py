"""
Vision Module Demo
Demonstrates object detection and color classification without robotic arm
"""

import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_module import VisionModule
from utils import get_statistics, print_statistics


def main():
    """Run vision demo."""
    print("\n" + "=" * 60)
    print("DOFBOT VISION MODULE DEMO")
    print("=" * 60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Show statistics")
    print("  'c' - Capture and save frame")
    print("=" * 60 + "\n")
    
    # Initialize vision module
    print("Initializing vision module...")
    vision = VisionModule(model_path='models/yolov8_dofbot.pt')
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("✅ Camera opened successfully")
    print("✅ Model loaded")
    print("\nStarting detection...\n")
    
    frame_count = 0
    capture_count = 0
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Detect objects
            detections = vision.detect_objects(frame, conf_threshold=0.5)
            
            # Classify by color
            color_bins = vision.classify_by_color(frame, detections)
            
            # Draw detections
            annotated = vision.draw_detections(frame, detections)
            
            # Add info overlay
            cv2.putText(annotated, f"Objects: {len(detections)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Add color distribution
            y_offset = 60
            for color, objects in color_bins.items():
                text = f"{color}: {len(objects)}"
                cv2.putText(annotated, text, 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
                y_offset += 25
            
            # Display frame
            cv2.imshow('DOFBOT Vision Demo', annotated)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            elif key == ord('s'):
                print("\n" + "-" * 50)
                print("CURRENT STATISTICS")
                print("-" * 50)
                stats = get_statistics(detections)
                print_statistics(stats)
                
                print("\nColor Distribution:")
                for color, objects in color_bins.items():
                    print(f"  {color}: {len(objects)} objects")
                print("-" * 50 + "\n")
            
            elif key == ord('c'):
                capture_count += 1
                filename = f"capture_{capture_count:03d}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"✅ Saved frame as {filename}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print(f"Total Frames Processed: {frame_count}")
        print(f"Captures Saved: {capture_count}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()