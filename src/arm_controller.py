import time
import numpy as np
from typing import Tuple, List, Dict


class ArmController:
    """Control DOFBOT 6-DOF robotic arm for desk tidying operations."""
    
    # Joint angle limits in degrees
    JOINT_LIMITS = {
        'j1': (-150, 150),   # Base rotation
        'j2': (-90, 90),     # Shoulder
        'j3': (-90, 90),     # Elbow
        'j4': (-90, 90),     # Wrist pitch
        'j5': (-90, 90),     # Wrist roll
        'j6': (-180, 180),   # Wrist yaw (gripper rotation)
    }
    
    # Safe neutral position
    HOME_POSITION = {
        'j1': 0, 'j2': 0, 'j3': 0,
        'j4': 0, 'j5': 0, 'j6': 0,
    }
    
    # Color to bin position mapping (8 bins for rainbow + neutrals)
    # Coordinates in cm: (x, y, z)
    BIN_POSITIONS = {
        # Rainbow bins (Row 1)
        'red': (20, -21, 5),      # Bin 1
        'orange': (20, -14, 5),   # Bin 2
        'yellow': (20, -7, 5),    # Bin 3
        'green': (20, 0, 5),      # Bin 4
        # Rainbow bins (Row 2)
        'blue': (20, 7, 5),       # Bin 5
        'indigo': (20, 14, 5),    # Bin 6
        'violet': (20, 21, 5),    # Bin 7
        # Neutral/misc bin (shared)
        'white': (20, 28, 5),     # Bin 8
        'black': (20, 28, 5),
        'cream': (20, 28, 5),
        'maroon': (20, 28, 5),
        'unknown': (20, 28, 5),
    }
    
    def __init__(self, port: str = None, baudrate: int = 115200):
        """
        Initialize DOFBOT arm controller.
        
        Args:
            port: Serial port for arm connection (e.g., 'COM3', '/dev/ttyUSB0')
            baudrate: Serial communication baud rate
        """
        self.port = port
        self.baudrate = baudrate
        self.current_position = self.HOME_POSITION.copy()
        self.gripper_open = True
        self.connected = False
        self.serial_conn = None
        
        print("🤖 ArmController initialized")
        print(f"   Port: {port or 'Not specified'}")
        print(f"   Baudrate: {baudrate}")
    
    def connect(self) -> bool:
        """
        Establish connection to DOFBOT via serial port.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Import serial library
            import serial
            
            if not self.port:
                print("⚠️  No port specified. Running in simulation mode.")
                self.connected = True
                return True
            
            # Establish serial connection
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1
            )
            
            self.connected = True
            print(f"✅ Connected to DOFBOT on {self.port}")
            
            # Initialize to home position
            time.sleep(0.5)
            self.move_home()
            
            return True
            
        except ImportError:
            print("⚠️  pyserial not installed. Running in simulation mode.")
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("⚠️  Running in simulation mode.")
            self.connected = True
            return False
    
    def disconnect(self):
        """Disconnect from DOFBOT."""
        if self.serial_conn:
            try:
                # Return to home before disconnecting
                self.move_home()
                time.sleep(0.5)
                self.serial_conn.close()
                print("✅ Disconnected from DOFBOT")
            except Exception as e:
                print(f"⚠️  Disconnect warning: {e}")
        
        self.connected = False
    
    def send_command(self, command: str) -> bool:
        """
        Send command string to DOFBOT.
        
        Args:
            command: Command string to send
            
        Returns:
            True if command sent successfully
        """
        if not self.connected:
            print("❌ Not connected to DOFBOT")
            return False
        
        try:
            if self.serial_conn:
                # Send actual command via serial
                self.serial_conn.write(f"{command}\n".encode())
                print(f"📤 Sent: {command}")
            else:
                # Simulation mode - just log
                print(f"🔶 [SIM] Command: {command}")
            
            return True
            
        except Exception as e:
            print(f"❌ Command failed: {e}")
            return False
    
    def validate_angles(self, angles: dict) -> bool:
        """
        Validate joint angles are within safe limits.
        
        Args:
            angles: Dictionary of joint angles
            
        Returns:
            True if all angles are valid
        """
        for joint, angle in angles.items():
            if joint in self.JOINT_LIMITS:
                min_angle, max_angle = self.JOINT_LIMITS[joint]
                if not (min_angle <= angle <= max_angle):
                    print(f"❌ Angle {angle}° out of range for {joint} "
                          f"(valid: {min_angle}° to {max_angle}°)")
                    return False
        return True
    
    def move_to_position(self, angles: dict, speed: float = 0.5, 
                        wait: bool = True) -> bool:
        """
        Move arm to specified joint angles.
        
        Args:
            angles: Dictionary of joint angles in degrees
            speed: Movement speed factor (0-1)
            wait: Whether to wait for movement to complete
            
        Returns:
            True if movement successful
        """
        # Validate angles
        if not self.validate_angles(angles):
            return False
        
        # Clamp speed
        speed = max(0.1, min(1.0, speed))
        
        # Build command string
        angle_str = ' '.join([f"{k}:{v}" for k, v in angles.items()])
        command = f"MOVE {angle_str} SPEED:{speed}"
        
        # Send command
        if self.send_command(command):
            # Update current position
            self.current_position.update(angles)
            
            # Wait for movement to complete
            if wait:
                delay = 1.0 / speed  # Longer delay for slower movements
                time.sleep(delay)
            
            return True
        
        return False
    
    def move_home(self) -> bool:
        """Move arm to safe home position."""
        print("🏠 Moving to home position...")
        return self.move_to_position(self.HOME_POSITION, speed=0.5)
    
    def open_gripper(self, delay: float = 0.5) -> bool:
        """
        Open gripper to release object.
        
        Args:
            delay: Time to wait for gripper to open (seconds)
            
        Returns:
            True if successful
        """
        if self.send_command("GRIPPER OPEN"):
            self.gripper_open = True
            print("✋ Gripper opened")
            time.sleep(delay)
            return True
        return False
    
    def close_gripper(self, delay: float = 0.5) -> bool:
        """
        Close gripper to grasp object.
        
        Args:
            delay: Time to wait for gripper to close (seconds)
            
        Returns:
            True if successful
        """
        if self.send_command("GRIPPER CLOSE"):
            self.gripper_open = False
            print("✊ Gripper closed")
            time.sleep(delay)
            return True
        return False
    
    def pick_object(self, position: Tuple[float, float, float], 
                   approach_height: float = 10) -> bool:
        """
        Pick up object from specified 3D position.
        
        Args:
            position: (x, y, z) coordinates in cm
            approach_height: Height above object for safe approach (cm)
            
        Returns:
            True if pick operation successful
        """
        x, y, z = position
        print(f"🎯 Picking object at ({x:.1f}, {y:.1f}, {z:.1f}) cm")
        
        # Step 1: Open gripper
        if not self.open_gripper():
            return False
        
        # Step 2: Move to approach position (above object)
        approach_pos = (x, y, z + approach_height)
        approach_angles = self.cartesian_to_angles(*approach_pos)
        print(f"   ↓ Approaching from above...")
        if not self.move_to_position(approach_angles, speed=0.4):
            return False
        
        # Step 3: Lower to object
        pick_angles = self.cartesian_to_angles(x, y, z)
        print(f"   ↓ Lowering to object...")
        if not self.move_to_position(pick_angles, speed=0.3):
            return False
        
        # Step 4: Close gripper
        print(f"   ✊ Grasping...")
        if not self.close_gripper():
            return False
        
        # Step 5: Lift object
        print(f"   ↑ Lifting object...")
        if not self.move_to_position(approach_angles, speed=0.4):
            return False
        
        print("✅ Object picked successfully")
        return True
    
    def place_object(self, position: Tuple[float, float, float],
                    approach_height: float = 10) -> bool:
        """
        Place object at specified 3D position.
        
        Args:
            position: (x, y, z) coordinates in cm
            approach_height: Height above placement for safe approach (cm)
            
        Returns:
            True if place operation successful
        """
        x, y, z = position
        print(f"📦 Placing object at ({x:.1f}, {y:.1f}, {z:.1f}) cm")
        
        # Step 1: Move to approach position (above placement)
        approach_pos = (x, y, z + approach_height)
        approach_angles = self.cartesian_to_angles(*approach_pos)
        print(f"   ↓ Approaching placement zone...")
        if not self.move_to_position(approach_angles, speed=0.4):
            return False
        
        # Step 2: Lower to placement position
        place_angles = self.cartesian_to_angles(x, y, z)
        print(f"   ↓ Lowering to bin...")
        if not self.move_to_position(place_angles, speed=0.3):
            return False
        
        # Step 3: Open gripper to release
        print(f"   ✋ Releasing object...")
        if not self.open_gripper():
            return False
        
        # Step 4: Lift away
        print(f"   ↑ Retracting...")
        if not self.move_to_position(approach_angles, speed=0.4):
            return False
        
        print("✅ Object placed successfully")
        return True
    
    def pick_and_place(self, pick_pos: Tuple[float, float, float],
                      place_pos: Tuple[float, float, float]) -> bool:
        """
        Complete pick-and-place operation from one location to another.
        
        Args:
            pick_pos: Position to pick from (x, y, z) in cm
            place_pos: Position to place at (x, y, z) in cm
            
        Returns:
            True if operation successful
        """
        print("=" * 60)
        print("🤖 PICK-AND-PLACE OPERATION STARTED")
        print("=" * 60)
        
        try:
            # Return to home first
            print("\n[1/5] Moving to home position...")
            if not self.move_home():
                return False
            time.sleep(0.3)
            
            # Pick object
            print("\n[2/5] Picking object...")
            if not self.pick_object(pick_pos):
                return False
            time.sleep(0.3)
            
            # Return to home with object
            print("\n[3/5] Returning to home...")
            if not self.move_home():
                return False
            time.sleep(0.3)
            
            # Place object
            print("\n[4/5] Placing object...")
            if not self.place_object(place_pos):
                return False
            time.sleep(0.3)
            
            # Final return to home
            print("\n[5/5] Final return to home...")
            if not self.move_home():
                return False
            
            print("=" * 60)
            print("✅ PICK-AND-PLACE COMPLETED SUCCESSFULLY")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"\n❌ Operation failed: {e}")
            print("🛑 Emergency stop triggered")
            self.emergency_stop()
            return False
    
    def cartesian_to_angles(self, x: float, y: float, z: float) -> dict:
        """
        Convert Cartesian coordinates to joint angles (inverse kinematics).
        
        This is a simplified IK solution. For production use, implement
        full DH parameter-based inverse kinematics for DOFBOT.
        
        Args:
            x, y, z: Position in cm
            
        Returns:
            Dictionary of joint angles in degrees
        """
        # Simplified inverse kinematics
        # Base rotation (J1) - rotate towards target
        j1 = int(np.degrees(np.arctan2(y, x)))
        
        # Calculate distance from base
        r = np.sqrt(x**2 + y**2)
        
        # Shoulder angle (J2) - rough approximation
        j2 = int(np.degrees(np.arctan2(z, r)))
        
        # Other joints - simplified
        j3 = -j2  # Elbow compensates shoulder
        j4 = 0    # Wrist pitch
        j5 = 0    # Wrist roll
        j6 = 0    # Gripper rotation
        
        angles = {
            'j1': j1, 'j2': j2, 'j3': j3,
            'j4': j4, 'j5': j5, 'j6': j6
        }
        
        # Clamp to limits
        for joint, angle in angles.items():
            min_a, max_a = self.JOINT_LIMITS[joint]
            angles[joint] = int(np.clip(angle, min_a, max_a))
        
        return angles
    
    def get_bin_position(self, color: str) -> Tuple[float, float, float]:
        """
        Get the 3D position of bin for specified color.
        
        Args:
            color: Color name (e.g., 'red', 'blue', 'green')
            
        Returns:
            (x, y, z) coordinates in cm
        """
        return self.BIN_POSITIONS.get(color.lower(), self.BIN_POSITIONS['unknown'])
    
    def emergency_stop(self) -> bool:
        """
        Emergency stop - halt all movement immediately.
        
        Returns:
            True if stop successful
        """
        print("🛑 EMERGENCY STOP ACTIVATED!")
        
        # Send emergency stop command
        success = self.send_command("ESTOP")
        
        # Open gripper for safety
        if self.gripper_open == False:
            self.open_gripper(delay=0.1)
        
        return success
    
    def test_movement(self):
        """Test basic arm movements."""
        print("\n" + "=" * 60)
        print("🧪 TESTING ARM MOVEMENTS")
        print("=" * 60)
        
        movements = [
            ("Home position", self.HOME_POSITION),
            ("Base rotation left", {'j1': -45, 'j2': 0, 'j3': 0, 'j4': 0, 'j5': 0, 'j6': 0}),
            ("Base rotation right", {'j1': 45, 'j2': 0, 'j3': 0, 'j4': 0, 'j5': 0, 'j6': 0}),
            ("Shoulder up", {'j1': 0, 'j2': 30, 'j3': 0, 'j4': 0, 'j5': 0, 'j6': 0}),
            ("Home position", self.HOME_POSITION),
        ]
        
        for name, angles in movements:
            print(f"\n➡️  {name}")
            self.move_to_position(angles, speed=0.5)
            time.sleep(0.5)
        
        # Test gripper
        print("\n🤲 Testing gripper...")
        self.open_gripper()
        time.sleep(0.5)
        self.close_gripper()
        time.sleep(0.5)
        self.open_gripper()
        
        print("\n✅ Movement test complete")


def test_arm():
    """Test arm controller functionality."""
    print("\n" + "=" * 60)
    print("🤖 DOFBOT ARM CONTROLLER TEST")
    print("=" * 60 + "\n")
    
    # Initialize arm (simulation mode if no port specified)
    arm = ArmController(port=None)  # Set to 'COM3' or '/dev/ttyUSB0' for real hardware
    
    # Connect
    if not arm.connect():
        print("❌ Connection failed")
        return
    
    # Test basic movements
    arm.test_movement()
    
    # Test pick and place
    print("\n" + "=" * 60)
    print("🧪 TESTING PICK-AND-PLACE")
    print("=" * 60)
    
    pick_position = (15, -10, 5)
    place_position = arm.get_bin_position('red')
    
    print(f"\nPick from: {pick_position}")
    print(f"Place at:  {place_position} (red bin)")
    
    arm.pick_and_place(pick_position, place_position)
    
    # Disconnect
    print("\n")
    arm.disconnect()
    
    print("\n✅ All tests complete!")


if __name__ == "__main__":
    test_arm()