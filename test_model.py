import cv2
from ultralytics import YOLO
import os

def test_on_image(model_path, image_path):
    """Test model on a single image"""
    print("\n" + "="*60)
    print("TESTING MODEL ON IMAGE")
    print("="*60)
    
    # Load model
    print(f"\n1. Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Load image
    print(f"2. Loading image from: {image_path}")
    image = cv2.imread(image_path)
    
    if image is None:
        print("❌ Error: Could not load image!")
        return
    
    print(f"3. Image size: {image.shape}")
    
    # Run detection
    print("\n4. Running detection...")
    results = model(image, conf=0.5)
    
    # Display results
    print("\n5. Detection Results:")
    print("-" * 60)
    
    for result in results:
        boxes = result.boxes
        print(f"   Found {len(boxes)} objects:")
        
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            
            print(f"   [{i+1}] {class_name}: {confidence:.2%}")
    
    # Show image with detections
    annotated = results[0].plot()
    
    cv2.imshow('Model Test - Press any key to close', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    output_path = "test_result.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"\n✅ Result saved to: {output_path}")


def test_on_webcam(model_path, conf_threshold=0.5):
    """Test model on webcam feed"""
    print("\n" + "="*60)
    print("TESTING MODEL ON WEBCAM")
    print("="*60)
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Take screenshot")
    print("  '+' - Increase confidence")
    print("  '-' - Decrease confidence")
    print("="*60 + "\n")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        return
    
    print("✅ Webcam opened")
    print(f"✅ Model loaded")
    print(f"\nConfidence threshold: {conf_threshold:.2f}\n")
    
    screenshot_count = 0
    frame_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame")
                break
            
            frame_count += 1
            
            # Run detection
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Get annotated frame
            annotated = results[0].plot()
            
            # Add info overlay
            info_text = f"Frame: {frame_count} | Conf: {conf_threshold:.2f} | Objects: {len(results[0].boxes)}"
            cv2.putText(annotated, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display detections
            for i, box in enumerate(results[0].boxes):
                class_name = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                
                text = f"{i+1}. {class_name}: {conf:.2%}"
                y_pos = 60 + (i * 25)
                cv2.putText(annotated, text, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Model Test - Webcam', annotated)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, annotated)
                print(f"📸 Screenshot saved: {filename}")
            
            elif key == ord('+'):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"Confidence threshold: {conf_threshold:.2f}")
            
            elif key == ord('-'):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"Confidence threshold: {conf_threshold:.2f}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total frames processed: {frame_count}")
        print(f"Screenshots saved: {screenshot_count}")
        print("="*60 + "\n")


def test_on_video(model_path, video_path):
    """Test model on a video file"""
    print("\n" + "="*60)
    print("TESTING MODEL ON VIDEO")
    print("="*60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = YOLO(model_path)
    
    # Open video
    print(f"Loading video from: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video!")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print("\nProcessing... Press 'q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        results = model(frame, conf=0.5, verbose=False)
        annotated = results[0].plot()
        
        # Add progress
        progress = f"Frame {frame_count}/{total_frames}"
        cv2.putText(annotated, progress, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Video Test - Press q to quit', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Processed {frame_count}/{total_frames} frames")


def quick_model_check(model_path):
    """Quick check if model loads correctly"""
    print("\n" + "="*60)
    print("QUICK MODEL CHECK")
    print("="*60)
    
    try:
        print(f"\n1. Loading model: {model_path}")
        model = YOLO(model_path)
        
        print("2. Model loaded successfully! ✅")
        print(f"3. Model type: {type(model)}")
        print(f"4. Number of classes: {len(model.names)}")
        print(f"5. Class names:")
        
        for i, name in model.names.items():
            print(f"   [{i}] {name}")
        
        print("\n✅ Model is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return False


def main():
    """Main testing menu"""
    print("\n" + "="*60)
    print("MODEL TESTING TOOL")
    print("="*60)
    
    # Get model path
    model_path = input("\nEnter model path (default: models/best.pt): ").strip()
    if not model_path:
        model_path = "models/best.pt"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("Please check the path and try again.")
        return
    
    # Quick check
    if not quick_model_check(model_path):
        return
    
    # Testing menu
    while True:
        print("\n" + "="*60)
        print("SELECT TEST MODE:")
        print("="*60)
        print("  1. Test on webcam (live detection)")
        print("  2. Test on image file")
        print("  3. Test on video file")
        print("  4. Quick model check")
        print("  0. Exit")
        print("="*60)
        
        choice = input("\nEnter choice (0-4): ").strip()
        
        if choice == '0':
            print("\nExiting...")
            break
        
        elif choice == '1':
            test_on_webcam(model_path)
        
        elif choice == '2':
            image_path = input("\nEnter image path: ").strip()
            if os.path.exists(image_path):
                test_on_image(model_path, image_path)
            else:
                print(f"❌ Error: Image not found at {image_path}")
        
        elif choice == '3':
            video_path = input("\nEnter video path: ").strip()
            if os.path.exists(video_path):
                test_on_video(model_path, video_path)
            else:
                print(f"❌ Error: Video not found at {video_path}")
        
        elif choice == '4':
            quick_model_check(model_path)
        
        else:
            print("Invalid choice. Please try again.")
    
    print("\n✅ Testing complete!\n")


if __name__ == "__main__":
    main()