import cv2
import time
import sys
import numpy as np
from typing import Tuple, List, Dict
from vision_module import VisionModule
from arm_controller import ArmController
from utils import (log_info, log_error, log_warning, save_operation_log,
                   get_statistics, print_statistics, Timer, format_time)


class DOFBOTSystem:
    """Main orchestrator for DOFBOT desk tidying system."""
    
    def __init__(self, model_path: str = 'models/best.pt',
                 arm_port: str = None, camera_id: int = 0):
        """
        Initialize complete DOFBOT system.
        
        Args:
            model_path: Path to trained YOLOv8 model (default: models/best.pt)
            arm_port: Serial port for arm (None for simulation)
            camera_id: Camera device ID (0 for default)
        """
        print("\n" + "=" * 70)
        print("🤖 DOFBOT SMART DESK-TIDYING ROBOT")
        print("=" * 70)
        print("Project: PDE3802 - AI in Robotics")
        print("Team: Kushmandaa, Kimberley, Leynah")
        print("=" * 70)
        print("\n📦 Your 8 Object Classes:")
        print("   1. AA Battery")
        print("   2. Charger Adapter")
        print("   3. Eraser")
        print("   4. Glue Stick")
        print("   5. Highlighter")
        print("   6. Pen")
        print("   7. Sharpener")
        print("   8. Stapler")
        print("=" * 70 + "\n")
        
        log_info("Initializing DOFBOT System...")
        
        # Initialize vision module
        print(f"📷 Loading vision module (model: {model_path})...")
        self.vision = VisionModule(model_path)
        
        # Initialize arm controller
        print("🤖 Loading arm controller...")
        self.arm = ArmController(port=arm_port)
        
        # Camera settings
        self.camera_id = camera_id
        
        # Operation statistics
        self.operation_count = 0
        self.successful_operations = 0
        self.failed_operations = 0
        
        log_info("✅ DOFBOT System initialized successfully")
    
    def connect_arm(self) -> bool:
        """
        Connect to robotic arm.
        
        Returns:
            True if connected successfully
        """
        log_info("Connecting to arm...")
        
        if self.arm.connect():
            log_info("✅ Arm connected successfully")
            return True
        else:
            log_error("❌ Failed to connect to arm")
            return False
    
    def scan_desk(self, conf_threshold: float = 0.5, 
                  show_preview: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        """
        Scan desk and detect all objects using camera.
        
        Args:
            conf_threshold: Confidence threshold for detection
            show_preview: Whether to show preview window
            
        Returns:
            Tuple of (frame, detections)
        """
        log_info("📷 Scanning desk...")
        
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            log_error("❌ Cannot open camera")
            return None, []
        
        # Capture frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            log_error("❌ Failed to capture frame")
            return None, []
        
        # Detect objects
        with Timer("Object detection", verbose=False):
            detections = self.vision.detect_objects(frame, conf_threshold)
        
        log_info(f"✅ Detected {len(detections)} objects")
        
        # Show preview if requested
        if show_preview and detections:
            annotated = self.vision.draw_detections(frame, detections, show_colors=True)
            cv2.imshow('Desk Scan', annotated)
            cv2.waitKey(2000)  # Show for 2 seconds
            cv2.destroyAllWindows()
        
        return frame, detections
    
    def classify_by_color(self, frame: np.ndarray, 
                         detections: List[Dict]) -> Dict[str, List]:
        """
        Organize detected objects by color for bin sorting.
        
        Args:
            frame: Camera frame
            detections: List of detections
            
        Returns:
            Dictionary mapping colors to lists of objects
        """
        log_info("🎨 Classifying objects by color...")
        
        color_bins = self.vision.classify_by_color(frame, detections)
        
        # Log distribution
        for color, objects in color_bins.items():
            log_info(f"   {color}: {len(objects)} object(s)")
        
        return color_bins
    
    def pick_and_sort_object(self, detection: Dict, color: str,
                            frame_shape: Tuple[int, int]) -> bool:
        """
        Pick detected object and place in appropriate color bin.
        
        Args:
            detection: Object detection info
            color: Detected color of object
            frame_shape: (height, width) of camera frame for coordinate conversion
            
        Returns:
            True if operation successful
        """
        self.operation_count += 1
        
        try:
            # Get object information
            class_name = detection['class_name']
            confidence = detection['confidence']
            center_x, center_y = detection['center']
            
            log_info(f"\n[Operation #{self.operation_count}]")
            log_info(f"Object: {class_name} ({color})")
            log_info(f"Confidence: {confidence:.2f}")
            
            # Get bin position for this color
            bin_pos = self.arm.get_bin_position(color)
            
            # Convert pixel coordinates to real-world coordinates
            # Simple scaling (needs calibration for real system)
            height, width = frame_shape
            # Assume camera sees 40cm x 30cm area
            x_cm = (center_x / width) * 40 - 20   # Center at 0
            y_cm = (center_y / height) * 30 - 15  # Center at 0
            z_cm = 5  # Assume objects are at desk level
            
            pick_pos = (x_cm, y_cm, z_cm)
            
            log_info(f"Pick position: {pick_pos}")
            log_info(f"Bin position: {bin_pos} ({color} bin)")
            
            # Perform pick and place operation
            with Timer(f"Pick-and-place {class_name}"):
                success = self.arm.pick_and_place(pick_pos, bin_pos)
            
            # Log operation
            if success:
                self.successful_operations += 1
                save_operation_log(
                    operation_name=f"pick_and_place_{class_name}",
                    success=True,
                    details={
                        'object': class_name,
                        'color': color,
                        'confidence': confidence,
                        'pick_position': pick_pos,
                        'bin_position': bin_pos
                    }
                )
                log_info(f"✅ Successfully sorted {class_name}")
            else:
                self.failed_operations += 1
                save_operation_log(
                    operation_name=f"pick_and_place_{class_name}",
                    success=False,
                    details={
                        'object': class_name,
                        'error': 'Movement failed'
                    }
                )
                log_error(f"❌ Failed to sort {class_name}")
            
            return success
            
        except Exception as e:
            self.failed_operations += 1
            log_error(f"❌ Error during pick and place: {e}")
            save_operation_log(
                operation_name=f"pick_and_place_{detection['class_name']}",
                success=False,
                details={'error': str(e)}
            )
            return False
    
    def tidy_desk(self, max_objects: int = None, conf_threshold: float = 0.5):
        """
        Main desk tidying operation - detect and sort all objects.
        
        Args:
            max_objects: Maximum number of objects to sort (None for all)
            conf_threshold: Detection confidence threshold
        """
        print("\n" + "=" * 70)
        print("🧹 STARTING DESK TIDYING OPERATION")
        print("=" * 70 + "\n")
        
        start_time = time.time()
        
        try:
            # Step 1: Move arm to home position
            log_info("[Step 1/5] Moving arm to home position...")
            self.arm.move_home()
            time.sleep(1)
            
            # Step 2: Scan desk
            log_info("[Step 2/5] Scanning desk for objects...")
            frame, detections = self.scan_desk(
                conf_threshold=conf_threshold,
                show_preview=False
            )
            
            if not detections:
                print("\n" + "=" * 70)
                print("✨ DESK IS ALREADY CLEAN!")
                print("=" * 70 + "\n")
                return
            
            # Step 3: Classify by color
            log_info("[Step 3/5] Classifying objects by color...")
            color_bins = self.classify_by_color(frame, detections)
            
            # Show statistics
            stats = get_statistics(detections)
            print_statistics(stats)
            
            # Step 4: Sort objects
            log_info("[Step 4/5] Sorting objects into bins...")
            print("\n" + "-" * 70)
            
            objects_processed = 0
            frame_shape = frame.shape[:2]
            
            for color, objects in color_bins.items():
                for detection in objects:
                    # Check if we've hit the limit
                    if max_objects and objects_processed >= max_objects:
                        log_info(f"\n⚠️  Reached maximum object limit ({max_objects})")
                        break
                    
                    # Sort this object
                    self.pick_and_sort_object(detection, color, frame_shape)
                    objects_processed += 1
                    
                    # Brief pause between operations
                    time.sleep(0.5)
                
                if max_objects and objects_processed >= max_objects:
                    break
            
            print("-" * 70)
            
            # Step 5: Return to home
            log_info("[Step 5/5] Returning arm to home position...")
            self.arm.move_home()
            
            # Calculate statistics
            elapsed_time = time.time() - start_time
            success_rate = (self.successful_operations / self.operation_count * 100 
                          if self.operation_count > 0 else 0)
            
            # Print final summary
            print("\n" + "=" * 70)
            print("✅ DESK TIDYING COMPLETE")
            print("=" * 70)
            print(f"📊 Operations: {self.operation_count}")
            print(f"✅ Successful: {self.successful_operations}")
            print(f"❌ Failed: {self.failed_operations}")
            print(f"📈 Success rate: {success_rate:.1f}%")
            print(f"⏱️  Total time: {format_time(elapsed_time)}")
            print("=" * 70 + "\n")
            
        except KeyboardInterrupt:
            log_warning("\n⚠️  Operation interrupted by user")
            self.arm.emergency_stop()
            print("\n🛑 Emergency stop activated")
            
        except Exception as e:
            log_error(f"\n❌ Unexpected error: {e}")
            self.arm.emergency_stop()
            print("\n🛑 Emergency stop activated")
    
    def demo_mode(self):
        """
        Demo mode - vision only without arm control.
        Shows real-time detection and color classification.
        """
        print("\n" + "=" * 70)
        print("🎥 DEMO MODE - Vision Only")
        print("=" * 70)
        print("Your 8 Objects:")
        print("  • aa_battery, charger_adapter, eraser, glue_stick")
        print("  • highlighter, pen, sharpener, stapler")
        print("\nControls:")
        print("  • Press 'q' to quit")
        print("  • Press 's' to show statistics")
        print("  • Press 'c' to capture frame")
        print("=" * 70 + "\n")
        
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            log_error("❌ Cannot open camera")
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                log_error("❌ Failed to capture frame")
                break
            
            # Detect objects
            detections = self.vision.detect_objects(frame, conf_threshold=0.5)
            
            # Classify by color
            color_bins = self.vision.classify_by_color(frame, detections)
            
            # Draw annotations
            annotated = self.vision.draw_detections(frame, detections, show_colors=True)
            
            # Add info overlay
            y_pos = 30
            cv2.putText(annotated, f"Objects: {len(detections)}", (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            y_pos += 35
            for color, objects in sorted(color_bins.items()):
                text = f"{color}: {len(objects)}"
                cv2.putText(annotated, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_pos += 25
            
            # Display
            cv2.imshow('DOFBOT Demo Mode', annotated)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                log_info("Exiting demo mode...")
                break
                
            elif key == ord('s'):
                # Show statistics
                stats = get_statistics(detections)
                print_statistics(stats)
                
            elif key == ord('c'):
                # Capture frame
                filename = f'demo_capture_{frame_count:04d}.jpg'
                cv2.imwrite(filename, annotated)
                log_info(f"💾 Saved: {filename}")
                frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        log_info("✅ Demo mode ended")
    
    def calibrate_camera(self):
        """
        Camera calibration mode to determine pixel-to-cm conversion.
        """
        print("\n" + "=" * 70)
        print("📐 CAMERA CALIBRATION MODE")
        print("=" * 70)
        print("Instructions:")
        print("  1. Place an object of known size (e.g., 10cm ruler)")
        print("  2. Press 'c' to capture and measure")
        print("  3. Press 'q' to quit")
        print("=" * 70 + "\n")
        
        cap = cv2.VideoCapture(self.camera_id)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Show frame with grid
            height, width = frame.shape[:2]
            cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 0), 1)
            cv2.line(frame, (0, height//2), (width, height//2), (0, 255, 0), 1)
            
            cv2.putText(frame, "Place calibration object and press 'c'", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                log_info("Calibration captured")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def shutdown(self):
        """Safely shutdown the system."""
        log_info("Shutting down DOFBOT system...")
        
        # Disconnect arm
        if self.arm.connected:
            self.arm.disconnect()
        
        log_info("✅ System shutdown complete")


def main():
    """Main entry point."""
    print("\n")
    
    # Initialize system with YOUR model path
    model_path = 'models/best.pt'
    system = DOFBOTSystem(model_path=model_path, arm_port=None)  # None = simulation
    
    # Menu
    while True:
        print("=" * 70)
        print("DOFBOT MAIN MENU")
        print("=" * 70)
        print("1. Tidy Desk (requires arm connection)")
        print("2. Demo Mode (vision only)")
        print("3. Calibrate Camera")
        print("4. Test Arm Movements")
        print("5. Exit")
        print("=" * 70)
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            # Connect arm and tidy desk
            if system.connect_arm():
                max_obj = input("Max objects to sort (Enter for all): ").strip()
                max_obj = int(max_obj) if max_obj else None
                system.tidy_desk(max_objects=max_obj)
            else:
                print("⚠️  Arm connection failed. Try demo mode instead.")
        
        elif choice == '2':
            # Demo mode
            system.demo_mode()
        
        elif choice == '3':
            # Calibration
            system.calibrate_camera()
        
        elif choice == '4':
            # Test arm
            if system.connect_arm():
                system.arm.test_movement()
            else:
                print("❌ Arm connection failed")
        
        elif choice == '5':
            # Exit
            system.shutdown()
            print("👋 Goodbye!\n")
            break
        
        else:
            print("❌ Invalid option")
        
        print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user")
        print("👋 Goodbye!\n")
        sys.exit(0)