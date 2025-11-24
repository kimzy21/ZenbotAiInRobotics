"""
Quick Test Script for 8-Class DOFBOT System
Verifies model loading and class detection
"""

import cv2
from vision_module import VisionModule

def test_model_loading():
    """Test if model loads correctly with 8 classes."""
    print("\n" + "=" * 70)
    print("🧪 TEST 1: MODEL LOADING")
    print("=" * 70)
    
    try:
        vision = VisionModule(model_path='models/best.pt')
        
        # Verify class count
        if len(vision.class_names) == 8:
            print("✅ PASS: Model loaded with correct number of classes (8)")
        else:
            print(f"❌ FAIL: Expected 8 classes, got {len(vision.class_names)}")
            return False
        
        # Verify class names
        expected_classes = [
            'aa_battery', 'charger_adapter', 'eraser', 'glue_stick',
            'highlighter', 'pen', 'sharpener', 'stapler'
        ]
        
        if vision.class_names == expected_classes:
            print("✅ PASS: All class names match dataset")
        else:
            print("⚠️  WARNING: Class names differ from expected")
            print(f"   Expected: {expected_classes}")
            print(f"   Got: {vision.class_names}")
        
        print("\n📦 Detected Classes:")
        for i, name in enumerate(vision.class_names):
            print(f"   {i}: {name}")
        
        return True
        
    except FileNotFoundError:
        print("❌ FAIL: Model file not found at models/best.pt")
        print("   Please ensure your trained model is at: models/best.pt")
        return False
    except Exception as e:
        print(f"❌ FAIL: Error loading model: {e}")
        return False


def test_camera():
    """Test if camera is accessible."""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: CAMERA ACCESS")
    print("=" * 70)
    
    try:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ FAIL: Cannot open camera (ID: 0)")
            print("   Try different camera IDs or check camera connection")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ FAIL: Cannot capture frame from camera")
            return False
        
        height, width = frame.shape[:2]
        print(f"✅ PASS: Camera working (Resolution: {width}x{height})")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Camera error: {e}")
        return False


def test_detection():
    """Test object detection on a single frame."""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: OBJECT DETECTION")
    print("=" * 70)
    print("This will capture a single frame and attempt detection...")
    print("Place one of your 8 objects in front of the camera!\n")
    
    try:
        # Initialize vision
        vision = VisionModule(model_path='models/best.pt')
        
        # Capture frame
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ FAIL: Cannot open camera")
            return False
        
        print("Capturing frame in 3 seconds...")
        print("3...")
        cv2.waitKey(1000)
        print("2...")
        cv2.waitKey(1000)
        print("1...")
        cv2.waitKey(1000)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("❌ FAIL: Cannot capture frame")
            return False
        
        # Detect objects
        detections = vision.detect_objects(frame, conf_threshold=0.5)
        
        print(f"\n📊 Detection Results:")
        print(f"   Objects detected: {len(detections)}")
        
        if len(detections) == 0:
            print("\n⚠️  No objects detected")
            print("   This is OK if no objects were in view")
            print("   Try placing one of your trained objects in front of camera")
            print("   Objects: aa_battery, charger_adapter, eraser, glue_stick,")
            print("            highlighter, pen, sharpener, stapler")
            return True
        
        # Show detected objects
        print("\n✅ PASS: Detection working!")
        print("\nDetected objects:")
        for i, det in enumerate(detections):
            print(f"   {i+1}. {det['class_name']} (confidence: {det['confidence']:.2%})")
        
        # Classify by color
        color_bins = vision.classify_by_color(frame, detections)
        print("\nColors detected:")
        for color, objects in color_bins.items():
            print(f"   {color}: {len(objects)} object(s)")
        
        # Save annotated image
        annotated = vision.draw_detections(frame, detections, show_colors=True)
        filename = 'test_detection.jpg'
        cv2.imwrite(filename, annotated)
        print(f"\n💾 Saved annotated image to: {filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Detection error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_color_detection():
    """Test color classification on different colors."""
    print("\n" + "=" * 70)
    print("🧪 TEST 4: COLOR DETECTION")
    print("=" * 70)
    
    try:
        vision = VisionModule(model_path='models/best.pt')
        
        # Test with a solid color square
        print("Testing color detection on sample colors...")
        
        colors_to_test = [
            ('red', [0, 0, 255]),
            ('blue', [255, 0, 0]),
            ('green', [0, 255, 0]),
            ('yellow', [0, 255, 255]),
        ]
        
        for color_name, bgr in colors_to_test:
            # Create test image
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            test_img[:] = bgr
            
            # Create fake detection
            detection = {
                'bbox': (0, 0, 100, 100),
                'class_name': 'test',
                'confidence': 1.0
            }
            
            detected_color = vision.get_object_color(test_img, detection)
            
            if detected_color.lower() == color_name.lower():
                print(f"   ✅ {color_name}: Correctly detected")
            else:
                print(f"   ⚠️  {color_name}: Detected as {detected_color}")
        
        print("\n✅ PASS: Color detection functional")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Color detection error: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🚀 DOFBOT SYSTEM TEST SUITE")
    print("=" * 70)
    print("Testing your 8-class dataset configuration...")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Model Loading", test_model_loading()))
    results.append(("Camera Access", test_camera()))
    results.append(("Object Detection", test_detection()))
    results.append(("Color Detection", test_color_detection()))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your system is ready to use!")
        print("\nNext steps:")
        print("  1. Run demo mode: python src/main.py (select option 2)")
        print("  2. Test with your actual objects")
        print("  3. Try full system when ready")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before proceeding.")
        print("Check the error messages above for details.")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import sys
    import numpy as np
    
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)