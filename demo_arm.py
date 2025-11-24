"""
Arm Controller Demo
Demonstrates robotic arm control without vision system
Tests movement, pick-and-place operations
"""
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from arm_controller import ArmController


def test_basic_movement(arm: ArmController):
    """Test basic arm movements."""
    print("\n" + "=" * 60)
    print("TEST 1: BASIC MOVEMENTS")
    print("=" * 60)
    
    # Move to home
    print("\n1. Moving to home position...")
    arm.move_home()
    time.sleep(2)
    
    # Test joint movements
    print("\n2. Testing individual joint movements...")
    
    # Base rotation
    print("   - Rotating base (J1)...")
    arm.move_to_position({'j1': 45})
    time.sleep(1)
    arm.move_to_position({'j1': -45})
    time.sleep(1)
    arm.move_to_position({'j1': 0})
    time.sleep(1)
    
    # Shoulder
    print("   - Moving shoulder (J2)...")
    arm.move_to_position({'j2': 30})
    time.sleep(1)
    arm.move_to_position({'j2': 0})
    time.sleep(1)
    
    # Elbow
    print("   - Moving elbow (J3)...")
    arm.move_to_position({'j3': 45})
    time.sleep(1)
    arm.move_to_position({'j3': 0})
    time.sleep(1)
    
    print("\n✅ Basic movement test complete")


def test_gripper(arm: ArmController):
    """Test gripper operations."""
    print("\n" + "=" * 60)
    print("TEST 2: GRIPPER OPERATIONS")
    print("=" * 60)
    
    print("\n1. Opening gripper...")
    arm.open_gripper()
    time.sleep(2)
    
    print("\n2. Closing gripper...")
    arm.close_gripper()
    time.sleep(2)
    
    print("\n3. Opening gripper...")
    arm.open_gripper()
    time.sleep(1)
    
    print("\n✅ Gripper test complete")


def test_pick_and_place(arm: ArmController):
    """Test pick and place operations."""
    print("\n" + "=" * 60)
    print("TEST 3: PICK AND PLACE OPERATIONS")
    print("=" * 60)
    
    # Define positions
    pick_pos = (15, 0, 5)  # cm
    place_pos_red = arm.get_bin_position('red')
    place_pos_blue = arm.get_bin_position('blue')
    place_pos_green = arm.get_bin_position('green')
    
    print("\n1. Pick and place to RED bin...")
    arm.pick_and_place(pick_pos, place_pos_red)
    time.sleep(2)
    
    print("\n2. Pick and place to BLUE bin...")
    arm.pick_and_place(pick_pos, place_pos_blue)
    time.sleep(2)
    
    print("\n3. Pick and place to GREEN bin...")
    arm.pick_and_place(pick_pos, place_pos_green)
    time.sleep(2)
    
    print("\n✅ Pick and place test complete")


def test_color_sorting_sequence(arm: ArmController):
    """Test sorting sequence for all colors."""
    print("\n" + "=" * 60)
    print("TEST 4: COLOR SORTING SEQUENCE")
    print("=" * 60)
    
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    pick_position = (15, 0, 5)
    
    for i, color in enumerate(colors, 1):
        print(f"\n{i}. Sorting to {color.upper()} bin...")
        bin_pos = arm.get_bin_position(color)
        arm.pick_and_place(pick_position, bin_pos)
        time.sleep(1)
    
    print("\n✅ Color sorting sequence complete")


def test_emergency_stop(arm: ArmController):
    """Test emergency stop functionality."""
    print("\n" + "=" * 60)
    print("TEST 5: EMERGENCY STOP")
    print("=" * 60)
    
    print("\n1. Starting movement...")
    arm.move_to_position({'j1': 45, 'j2': 30})
    
    print("\n2. EMERGENCY STOP!")
    arm.emergency_stop()
    
    time.sleep(2)
    
    print("\n3. Recovering - moving to home...")
    arm.move_home()
    
    print("\n✅ Emergency stop test complete")


def interactive_mode(arm: ArmController):
    """Interactive mode for manual testing."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("\nCommands:")
    print("  h - Move to home")
    print("  o - Open gripper")
    print("  c - Close gripper")
    print("  p - Pick from default position")
    print("  1-8 - Place in bin (1=red, 2=orange, ...)")
    print("  e - Emergency stop")
    print("  q - Quit")
    print("=" * 60 + "\n")
    
    default_pick_pos = (15, 0, 5)
    
    while True:
        cmd = input("\nEnter command: ").strip().lower()
        
        if cmd == 'q':
            print("Exiting interactive mode...")
            break
        
        elif cmd == 'h':
            print("Moving to home...")
            arm.move_home()
        
        elif cmd == 'o':
            print("Opening gripper...")
            arm.open_gripper()
        
        elif cmd == 'c':
            print("Closing gripper...")
            arm.close_gripper()
        
        elif cmd == 'p':
            print(f"Picking from {default_pick_pos}...")
            arm.pick_object(default_pick_pos)
        
        elif cmd in '12345678':
            colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'white']
            color = colors[int(cmd) - 1]
            bin_pos = arm.get_bin_position(color)
            print(f"Placing in {color} bin at {bin_pos}...")
            arm.place_object(bin_pos)
        
        elif cmd == 'e':
            print("EMERGENCY STOP!")
            arm.emergency_stop()
        
        else:
            print("Unknown command. Type 'q' to quit.")


def main():
    """Run arm controller demo."""
    print("\n" + "=" * 60)
    print("DOFBOT ARM CONTROLLER DEMO")
    print("=" * 60)
    
    # Check if user wants to connect to real hardware
    print("\nDo you have the DOFBOT arm connected? (y/n): ", end='')
    response = input().strip().lower()
    
    if response == 'y':
        print("\nEnter serial port (e.g., COM3, /dev/ttyUSB0): ", end='')
        port = input().strip()
        
        # Initialize arm controller
        print(f"\nInitializing arm on {port}...")
        arm = ArmController(port=port)
        
        if not arm.connect():
            print("❌ Failed to connect to arm")
            print("Running in simulation mode...")
    else:
        print("\nRunning in SIMULATION MODE (no hardware required)")
        arm = ArmController()
    
    print("\n✅ Arm controller initialized")
    
    # Menu
    while True:
        print("\n" + "=" * 60)
        print("SELECT TEST:")
        print("=" * 60)
        print("  1. Basic Movements")
        print("  2. Gripper Operations")
        print("  3. Pick and Place")
        print("  4. Color Sorting Sequence")
        print("  5. Emergency Stop")
        print("  6. Interactive Mode")
        print("  7. Run All Tests")
        print("  0. Exit")
        print("=" * 60)
        
        choice = input("\nEnter choice (0-7): ").strip()
        
        try:
            if choice == '0':
                print("\nExiting...")
                break
            
            elif choice == '1':
                test_basic_movement(arm)
            
            elif choice == '2':
                test_gripper(arm)
            
            elif choice == '3':
                test_pick_and_place(arm)
            
            elif choice == '4':
                test_color_sorting_sequence(arm)
            
            elif choice == '5':
                test_emergency_stop(arm)
            
            elif choice == '6':
                interactive_mode(arm)
            
            elif choice == '7':
                print("\n🚀 RUNNING ALL TESTS...\n")
                test_basic_movement(arm)
                test_gripper(arm)
                test_pick_and_place(arm)
                test_color_sorting_sequence(arm)
                test_emergency_stop(arm)
                
                print("\n" + "=" * 60)
                print("✅ ALL TESTS COMPLETE")
                print("=" * 60)
            
            else:
                print("Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            arm.emergency_stop()
        
        except Exception as e:
            print(f"\n❌ Error during test: {e}")
            arm.emergency_stop()
    
    # Cleanup
    if arm.connected:
        print("\nMoving to home position...")
        arm.move_home()
        arm.disconnect()
    
    print("\n✅ Demo complete\n")


if __name__ == "__main__":
    main()