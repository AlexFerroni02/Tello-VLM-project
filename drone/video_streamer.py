"""
Tello Video Streamer Module.
Provides the TelloVideoStreamer class which manages the H.264 camera stream,
frame retrieval, and downscaling for VLM input using OpenCV.
"""

import cv2
from typing import Optional, Tuple
import numpy as np
from djitellopy import Tello

class TelloVideoStreamer:
    """
    Manages the decoding and scaling of H.264 video streams sent from the Tello camera.
    """
    def __init__(self, tello_instance: Tello):
        """
        Initializes the video streamer by binding it to an active Tello connection.
        
        Parameters:
            tello_instance (Tello): An active, connected djitellopy Tello object.
            
        Returns:
            None
        """
        self.tello = tello_instance
        self.frame_reader = None
        self.is_streaming = False

    def start(self) -> bool:
        """
        Commands the Tello to start streaming video over UDP port 11111,
        and initializes the PyAV frame reader background thread.
        
        Parameters:
            None
            
        Returns:
            bool: True if video stream started successfully, False otherwise.
            
        Raises:
            Exception: Propagates stream start errors from djitellopy socket bindings.
        """
        print("Starting video stream transport from drone...")
        try:
            # Send the command to turn on video streaming
            self.tello.streamon()
            
            # Retrieve the frame reader. djitellopy starts a background thread 
            # that continuously decodes H.264 packets into numpy RGB arrays.
            self.frame_reader = self.tello.get_frame_read()
            self.is_streaming = True
            print("Video stream listener initialized.")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize video stream. Details: {e}")
            self.is_streaming = False
            return False

    def get_raw_frame(self) -> Optional[np.ndarray]:
        """
        Gets the raw RGB frame from the Tello frame reader thread.
        
        Parameters:
            None
            
        Returns:
            Optional[np.ndarray]: A NumPy array of shape (720, 1280, 3) containing 
                                  the RGB image, or None if the buffer is empty.
        """
        if not self.is_streaming or self.frame_reader is None:
            return None
            
        frame = self.frame_reader.frame
        # Verify that the frame is not empty and has content
        if frame is None or frame.size == 0:
            return None
            
        return frame

    def get_vlm_resized_frame(self, target_w: int, target_h: int) -> Optional[np.ndarray]:
        """
        Retrieves the current frame and scales it down to the target VLM dimensions.
        Uses INTER_AREA interpolation, which is optimal for downsampling images 
        without introducing high-frequency aliasing artifacts (sgranature).
        
        Parameters:
            target_w (int): The target width of the output image in pixels.
            target_h (int): The target height of the output image in pixels.
            
        Returns:
            Optional[np.ndarray]: Resized RGB NumPy array of shape (target_h, target_w, 3) 
                                  or None if the raw frame reader is not ready.
        """
        raw_frame = self.get_raw_frame()
        if raw_frame is None:
            return None
            
        # OpenCV resize expects (width, height) tuple
        resized = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        return resized

    def stop(self):
        """
        Safely stops the video stream to release PyAV threads and decoder memory.
        
        Parameters:
            None
            
        Returns:
            None
        """
        print("Stopping video stream...")
        if self.is_streaming:
            try:
                self.tello.streamoff()
            except Exception as e:
                print(f"Warning during stream shutdown: {e}")
        self.is_streaming = False
        self.frame_reader = None
        print("Video stream stopped.")
