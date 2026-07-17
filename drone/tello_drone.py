"""
Tello Drone Wrapper Module.
Provides the high-level TelloDrone class which acts as the unified API
for connecting, reading telemetry, streaming video, and executing flight commands.
"""

from typing import Dict, Any, Optional
import numpy as np
from djitellopy import Tello
from config import TELLO_IP, VLM_INPUT_WIDTH, VLM_INPUT_HEIGHT
from drone.video_streamer import TelloVideoStreamer

class TelloDrone:
    """
    Unified controller class for the DJI Tello drone.
    Encapsulates SDK commands, telemetry parsing, and H.264 video decoding.
    """
    
    def __init__(self):
        """
        Initializes the TelloDrone controller interface.
        Creates the underlying Tello connection client but does not connect immediately.
        """
        self.tello: Tello = Tello(host=TELLO_IP)
        self.streamer: Optional[TelloVideoStreamer] = None
        self.is_connected: bool = False

    def connect(self) -> bool:
        """
        Establishes UDP connection with the Tello drone and triggers SDK mode.
        
        Parameters:
            None
            
        Returns:
            bool: True if connection handshake succeeded, False otherwise.
            
        Raises:
            Exception: Propagates network socket errors from djitellopy connection checks.
        """
        print(f"Initiating handshake to Tello at {TELLO_IP}...")
        try:
            self.tello.connect()
            self.is_connected = True
            print("Tello connection handshake completed successfully.")
            return True
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to connect to Tello. Details: {e}")
            self.is_connected = False
            return False

    def start_video(self) -> bool:
        """
        Starts the H.264 video streamer and initializes PyAV frame buffers.
        
        Parameters:
            None
            
        Returns:
            bool: True if video stream started successfully, False otherwise.
        """
        if not self.is_connected:
            print("Cannot start video: Drone is not connected.")
            return False
            
        self.streamer = TelloVideoStreamer(self.tello)
        return self.streamer.start()

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Queries raw sensor readouts from the Tello and packages them into a telemetry dictionary.
        
        Parameters:
            None
            
        Returns:
            Dict[str, Any]: A dictionary containing sensor keys:
                - "battery" (int): Battery percentage (0-100)
                - "temperature" (int): Highest board temperature in Celsius
                - "height" (int): Time-of-flight infrared distance from ground in cm
                - "barometer" (float): Air pressure height equivalent in cm
                - "flight_time" (int): Total active motor run time in seconds
        """
        if not self.is_connected:
            return {"battery": 0, "temperature": 0, "height": 0, "barometer": 0.0, "flight_time": 0}
            
        try:
            return {
                "battery": self.tello.get_battery(),
                "temperature": self.tello.get_highest_temperature(),
                "height": self.tello.get_height(),
                "barometer": self.tello.get_barometer(),
                "flight_time": self.tello.get_flight_time()
            }
        except Exception as e:
            print(f"Warning: Failed to fetch active telemetry from Tello. Details: {e}")
            return {"battery": -1, "temperature": -1, "height": -1, "barometer": -1.0, "flight_time": -1}

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Gets the current raw RGB camera frame (1280x720).
        
        Parameters:
            None
            
        Returns:
            Optional[np.ndarray]: OpenCV-compatible RGB matrix or None if frame is empty.
        """
        if self.streamer is None:
            return None
        return self.streamer.get_raw_frame()

    def get_vlm_frame(self) -> Optional[np.ndarray]:
        """
        Gets the current frame downscaled to the dimensions specified in the config.py.
        
        Parameters:
            None
            
        Returns:
            Optional[np.ndarray]: Resized RGB matrix or None if frame is empty.
        """
        if self.streamer is None:
            return None
        return self.streamer.get_vlm_resized_frame(VLM_INPUT_WIDTH, VLM_INPUT_HEIGHT)

    def takeoff(self) -> bool:
        """
        Commands the drone to auto-takeoff. 
        Drone will hover at approximately 80cm-100cm from the ground.
        
        Parameters:
            None
            
        Returns:
            bool: True if takeoff command completed successfully, False otherwise.
        """
        if not self.is_connected:
            print("Cannot takeoff: Drone is not connected.")
            return False
            
        # Quick safety check
        bat = self.tello.get_battery()
        if bat < 15:
            print(f"Takeoff aborted: Battery level {bat}% is below safety threshold (15%).")
            return False
            
        print("Sending takeoff command...")
        try:
            self.tello.takeoff()
            print("Takeoff completed. Drone is hovering.")
            return True
        except Exception as e:
            print(f"Error during takeoff execution: {e}")
            return False

    def land(self) -> bool:
        """
        Commands the drone to execute automatic landing.
        
        Parameters:
            None
            
        Returns:
            bool: True if landing command completed successfully, False otherwise.
        """
        if not self.is_connected:
            print("Cannot land: Drone is not connected.")
            return False
            
        print("Sending land command...")
        try:
            self.tello.land()
            print("Landing completed.")
            return True
        except Exception as e:
            print(f"Error during landing execution: {e}")
            return False

    def move_forward(self, distance: int) -> bool:
        """
        Commands the drone to fly forward by a specified distance.
        
        Parameters:
            distance (int): Distance in centimeters (between 20 and 500).
            
        Returns:
            bool: True if command executed successfully, False otherwise.
        """
        if not self.is_connected:
            return False
        print(f"Sending movement command: Move Forward {distance} cm")
        try:
            self.tello.move_forward(distance)
            return True
        except Exception as e:
            print(f"Error during forward movement: {e}")
            return False

    def move_backward(self, distance: int) -> bool:
        """
        Commands the drone to fly backward by a specified distance.
        
        Parameters:
            distance (int): Distance in centimeters (between 20 and 500).
            
        Returns:
            bool: True if command executed successfully, False otherwise.
        """
        if not self.is_connected:
            return False
        print(f"Sending movement command: Move Backward {distance} cm")
        try:
            self.tello.move_backward(distance)
            return True
        except Exception as e:
            print(f"Error during backward movement: {e}")
            return False

    def move_left(self, distance: int) -> bool:
        """
        Commands the drone to fly sideways to the left by a specified distance.
        
        Parameters:
            distance (int): Distance in centimeters (between 20 and 500).
            
        Returns:
            bool: True if command executed successfully, False otherwise.
        """
        if not self.is_connected:
            return False
        print(f"Sending movement command: Move Left {distance} cm")
        try:
            self.tello.move_left(distance)
            return True
        except Exception as e:
            print(f"Error during left movement: {e}")
            return False

    def move_right(self, distance: int) -> bool:
        """
        Commands the drone to fly sideways to the right by a specified distance.
        
        Parameters:
            distance (int): Distance in centimeters (between 20 and 500).
            
        Returns:
            bool: True if command executed successfully, False otherwise.
        """
        if not self.is_connected:
            return False
        print(f"Sending movement command: Move Right {distance} cm")
        try:
            self.tello.move_right(distance)
            return True
        except Exception as e:
            print(f"Error during right movement: {e}")
            return False

    def move_up(self, distance: int) -> bool:
        """
        Commands the drone to increase its altitude by a specified distance.
        
        Parameters:
            distance (int): Distance in centimeters (between 20 and 500).
            
        Returns:
            bool: True if command executed successfully, False otherwise.
        """
        if not self.is_connected:
            return False
        print(f"Sending movement command: Move Up {distance} cm")
        try:
            self.tello.move_up(distance)
            return True
        except Exception as e:
            print(f"Error during ascend: {e}")
            return False

    def move_down(self, distance: int) -> bool:
        """
        Commands the drone to decrease its altitude by a specified distance.
        
        Parameters:
            distance (int): Distance in centimeters (between 20 and 500).
            
        Returns:
            bool: True if command executed successfully, False otherwise.
        """
        if not self.is_connected:
            return False
        print(f"Sending movement command: Move Down {distance} cm")
        try:
            self.tello.move_down(distance)
            return True
        except Exception as e:
            print(f"Error during descend: {e}")
            return False

    def rotate_clockwise(self, angle: int) -> bool:
        """
        Commands the drone to rotate clockwise (Yaw Right) by a specified angle.
        
        Parameters:
            angle (int): Rotation angle in degrees (between 1 and 360).
            
        Returns:
            bool: True if rotation executed successfully, False otherwise.
        """
        if not self.is_connected:
            return False
        print(f"Sending rotation command: Rotate Clockwise {angle} degrees")
        try:
            self.tello.rotate_clockwise(angle)
            return True
        except Exception as e:
            print(f"Error during clockwise rotation: {e}")
            return False

    def rotate_counter_clockwise(self, angle: int) -> bool:
        """
        Commands the drone to rotate counter-clockwise (Yaw Left) by a specified angle.
        
        Parameters:
            angle (int): Rotation angle in degrees (between 1 and 360).
            
        Returns:
            bool: True if rotation executed successfully, False otherwise.
        """
        if not self.is_connected:
            return False
        print(f"Sending rotation command: Rotate Counter-Clockwise {angle} degrees")
        try:
            self.tello.rotate_counter_clockwise(angle)
            return True
        except Exception as e:
            print(f"Error during counter-clockwise rotation: {e}")
            return False

    def disconnect(self):
        """
        Safely stops background video threads and terminates UDP connections.
        
        Parameters:
            None
            
        Returns:
            None
        """
        print("Releasing Tello Drone resources...")
        if self.streamer is not None:
            self.streamer.stop()
        self.tello.end()
        self.is_connected = False
        print("Tello Drone resources disconnected.")
