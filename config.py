"""
Global configuration settings for the Tello-VLM project.
Centralizes all drone network parameters, image resolutions, and GUI settings.
"""

# Default IP address of the DJI Tello access point
TELLO_IP = "192.168.10.1"

# --- VIDEO RESIZING SETTINGS ---
# VLM input dimensions. Moondream2 works best with 378x378.
VLM_INPUT_WIDTH = 378
VLM_INPUT_HEIGHT = 378

# --- GUI COCKPIT SETTINGS ---
COCKPIT_WINDOW_TITLE = "Tello Autopilot Console"
COCKPIT_PANEL_WIDTH = 400       # Width of the telemetry/VLM status panel on the right (pixels)
COCKPIT_VIDEO_WIDTH = 960       # Width to display the video on the left (downscaled from 1280 to fit screens)
COCKPIT_VIDEO_HEIGHT = 540      # Height to display the video on the left (downscaled from 720 to keep 16:9)

# --- VLM CORE SETTINGS & OPTIMIZATIONS ---
VLM_MODEL_ID = "vikhyatk/moondream2"
VLM_REVISION = "2024-08-26"    # Pinned stable release version
VLM_CACHE_DIR = "./vlm_cache"  # Local project folder for storing cached weights offline
VLM_SERVER_URL = "http://localhost:8080/v1/chat/completions"  # Local llama.cpp endpoint for SmolVLM
