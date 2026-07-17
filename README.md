# DJI Tello VLM Autopilot - Educational Hub & Documentation

This repository contains an autonomous edge-controlled system for the DJI Tello drone, integrated with a Vision-Language Model (VLM) optimized for consumer GPUs (like the GTX 1650 4GB).

---

## Table of Lessons

1. [Lesson 1: Virtual Environments & IDE Configuration](#lesson-1-virtual-environments--ide-configuration)
2. [Lesson 2: UDP Networking & Multithreading in Drones](#lesson-2-udp-networking--multithreading-in-drones)
3. [Lesson 3: Computer Vision downscaling, BGR channels, and Canvas UI](#lesson-3-computer-vision-downscaling-bgr-channels-and-canvas-ui)
4. [Lesson 4: Object-Oriented Programming & Dependency Injection](#lesson-4-object-oriented-programming--dependency-injection)
5. [Lesson 5: Robotic Controls & Interactive Piloting](#lesson-5-robotic-controls--interactive-piloting)
6. [Lesson 6: Decoupled Multi-Modal Server (Llama.cpp & SmolVLM)](#lesson-6-decoupled-multi-modal-server-llamacpp--smolvlm)
7. [Quick Start & Verification Commands](#quick-start--verification-commands)

---

## Lesson 1: Virtual Environments & IDE Configuration

### Why use a Virtual Environment (`venv`)?
In Python, installing libraries globally (system-wide) causes version conflicts over time. For example, Project A might require `numpy v1.20` while Project B requires `numpy v2.4`. 
A virtual environment (`venv`) is a self-contained directory containing:
*   A localized copy of the Python interpreter (`venv/Scripts/python.exe`).
*   A private `site-packages` directory where libraries are installed without affecting the rest of your system.

### Solving the IDE "Red Line" Error
When you open a python script, your IDE (VS Code, Cursor, PyCharm) uses its default Python analyzer. If the IDE points to the system-wide Python, it cannot see the libraries installed inside your project's `venv` folder, showing red squiggly lines on imports like `from djitellopy import Tello`.
*   **Solution**: Press `Ctrl+Shift+P` -> `Python: Select Interpreter` -> choose the Python executable inside your `./venv/Scripts/` folder. This links the IDE's static analysis tool to the correct dependencies folder.

### Python 3.13 & CUDA 12.4 Compatibility
Python 3.13.1 is very recent. Stable PyTorch packages for older CUDA versions (like CUDA 12.1) do not have pre-built binaries (wheels) for Windows Python 3.13 on the default index.
*   **Resolution**: By pointing the installation to the PyTorch CUDA 12.4 repository (`https://download.pytorch.org/whl/cu124`), we get official pre-compiled wheels.
*   **Driver support**: The GTX 1650 Ti GPU supports CUDA 12.x. To use it, the system's NVIDIA Graphics Driver must be version `551.61` or higher. Running `nvidia-smi` in the command prompt reveals the installed driver version.

---

## Lesson 2: UDP Networking & Multithreading in Drones

### UDP (User Datagram Protocol) vs TCP
Most web systems use TCP (Transmission Control Protocol) which guarantees that all packets arrive in order and undamaged. If a packet is lost, TCP pauses the stream and retransmits it.
For robotics and drone streaming, **UDP** is preferred:
*   UDP is connectionless and sends packets immediately without waiting for handshakes or error checks.
*   If a video frame is lost during flight, we do not want to freeze the video to retransmit it. We prefer to skip it and show the next frame immediately to maintain real-time telemetry.

### Tello UDP Ports Architecture
The DJI Tello IP address is hardcoded in its firmware to `192.168.10.1`. It opens three UDP ports for parallel communication:
1.  **Port 8889 (Commands)**: The PC sends ASCII text strings (e.g., `takeoff`, `land`, `forward 50`) and receives ASCII replies (e.g., `ok`, `error`).
2.  **Port 8890 (State/Telemetry)**: The drone continuously broadcasts its telemetry parameters (battery, temperature, barometer) to this port.
3.  **Port 11111 (Video)**: The front camera streams raw H.264 video packets to this port.

```
                  +-------------------------------------------------------+
                  |                      PC WINDOWS                       |
                  |                                                       |
                  |  [Thread A] listens on UDP Port 8890 (Telemetry)     |
                  |  -> Writes battery, temp, and height into RAM.        |
                  |                                                       |
                  |  [Thread B] listens on UDP Port 11111 (Video feed)    |
                  |  -> Decodes H.264 packets into numpy arrays in RAM.   |
                  |                                                       |
+------------+    |  [Main Thread (Your Script)]                          |
|   TELLO    |===>|  Reads telemetry & frames from RAM concurrently       |
| (192.168.1) |    |  without sending new network requests.                |
+------------+    +-------------------------------------------------------+
```

### Telemetry Variables & Hardware Safety
*   **CPU Temperature**: The Tello is cooled passively by wind from the propellers. When idling on a desk, it has no airflow. If its temperature exceeds 80°C, it will trigger an automatic emergency shutdown to prevent processor damage. Monitoring `tello.get_highest_temperature()` is critical.
*   **Altitude Sensors**: The Tello has two altitude systems:
    *   *Infrared/ToF (`get_height()`)*: Extremely precise (in centimeters) from the ground up to 3 meters. It starts at `0`.
    *   *Barometer (`get_barometer()`)*: Measures air pressure to estimate height. It starts at sea-level relative height (e.g., 100 meters equivalent) and is subject to local room pressure variations.

---

## Lesson 3: Computer Vision downscaling, BGR channels, and Canvas UI

### Downscaling for Edge VLMs (4GB VRAM constraint)
The Tello camera outputs a **1280x720 (720p)** H.264 video stream. Feeding a 1280x720 image into a Vision-Language Model uses massive amounts of activation memory, triggering an out-of-memory error (OOM) on a 4GB GPU.
*   **Resolution**: We downscale the frame to `378x378` (Moondream2 size) or `448x448` (Qwen2-VL size) before running inference.
*   **Interpolation**: We use OpenCV `cv2.resize` with `cv2.INTER_AREA`. While bilinear or bicubic interpolation is faster, `INTER_AREA` is mathematically optimal for shrinking images (downsampling) because it avoids pixel aliasing (jagged edges), providing a cleaner image for the VLM.

### BGR vs RGB Color Formats
OpenCV stores images in **BGR** (Blue, Green, Red) order. This goes back to early CCD camera manufacturers in the late 1990s.
Most modern AI packages, pillow, and the Tello stream use **RGB** (Red, Green, Blue) order.
If you attempt to display Tello frames in OpenCV without conversion, colors will be inverted (red faces will look blue). We must run:
`cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)`

### How the Cockpit GUI is Drawn
OpenCV does not have HTML-like layout elements. Everything is drawn on a raw matrix (NumPy array) of pixels.
To display the video next to the telemetry text, we use **Horizontal Stacking (`np.hstack`)**:
1.  Resize the live video to `960x540`.
2.  Create a blank gray panel (`np.zeros`) of size `400x540`.
3.  Draw text and shapes (like the battery bar) onto the panel using `cv2.putText` and `cv2.line`.
4.  Join them horizontally: `canvas = np.hstack((video, panel))`. This creates a unified `1360x540` console image.

---

## Lesson 4: Object-Oriented Programming & Dependency Injection

To avoid creating a **"God Object"** (a single class that does everything and becomes impossible to test/maintain), we divide our project into modular, independent classes:

1.  **`TelloVideoStreamer` (video_streamer.py)**: Manages camera sockets and PyAV frame buffers.
2.  **`TelloDrone` (tello_drone.py)**: A high-level interface class (wrapper) that ties the connection, flight commands, and video streamer together into a single, clean API.

### Dependency Injection Pattern
Instead of `TelloVideoStreamer` creating its own connection socket, we connect to the drone first, and then pass the connected `Tello` instance to the streamer:
`streamer = TelloVideoStreamer(tello)`
This ensures we only open **one single UDP connection socket** on port 8889, preventing conflict errors between classes.

---

## Lesson 5: Robotic Controls & Interactive Piloting

### Blocking vs Non-Blocking Control
When commanding drone movements, the Tello SDK supports two types of interactions:
1.  **Blocking commands (`tello.move_...`)**: These calls tell the drone to move a specific distance (e.g. forward 40cm) and block execution until it finishes.
2.  **Non-Blocking commands (`tello.send_rc_control`)**: These commands set continuous speed velocities on 4 channels (pitch, roll, throttle, yaw) and return immediately. The drone flies at that speed until a new command is received.
    *   *Web App*: Our Web Autopilot dashboard uses non-blocking RC velocity commands to allow smooth, real-time manual control and VLM co-pilot telemetry sync without video stream freezing.

### Fail-Safe Landing Design
In robotics, network drops or script crashes can leave a drone stranded in flight. We implement a **fail-safe block** utilizing Python's `finally` constructor:
```python
try:
    # Flight execution loop...
finally:
    # This block executes NO MATTER WHAT (user interrupts, socket exceptions, crashes)
    if drone.is_connected:
        drone.land()  # Force safe automatic landing
```

### Manual Key Mapping Table
When focused on the browser Cockpit HUD page, the web interface captures keyboard events to steer the drone:

| Key | Keyboard Key | Drone Action | Direction / Speed |
| :--- | :--- | :--- | :--- |
| **`t`** | T Key | **Takeoff** | Ascend to ~80cm |
| **`Spacebar`** | Space Key | **LAND (FAIL-SAFE)** | Execute Auto-landing |
| **`w` / `s`** | W / S Keys | Pitch Forward / Backward | Forward / Backward Translation |
| **`a` / `d`** | A / D Keys | Roll Left / Right | Sideways Translation |
| **`i` / `k`** | I / K Keys | Throttle Up / Down | Vertical Translation |
| **`j` / `l`** | J / L Keys | Yaw Left / Right | Rotation / Turn |

---

## Quick Start & Verification Commands

### 1. Requirements Installation
```powershell
# Install PyTorch with CUDA 12.4 first
.\venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install the rest of the packages (including requests for VLM server)
.\venv\Scripts\python -m pip install -r requirements.txt requests
```

### 2. Setup VLM Server (Llama.cpp + SmolVLM)
To automatically download the latest precompiled `llama.cpp` CUDA binaries and the SmolVLM GGUF models:
```powershell
.\venv\Scripts\python setup_llama_cpp.py
```
*(This creates the `llama_server` and `llama_models` folders and generates a launch script).*

### 3. Launch the VLM Server
Start the local GPU-accelerated GGUF inference engine:
```powershell
.\run_llama_server.bat
```
*(Keep this window open in the background!)*

### 4. Run the Autopilot Web Cockpit
Launches the Flask web server and drone control interface:
```powershell
.\venv\Scripts\python -m drone.web_server
```
Open `http://localhost:5000` to fly!

### 5. Run the Webcam Diagnostic Test (Webcam instead of Drone)
If you want to test VLM inference locally using your PC webcam:
```powershell
.\venv\Scripts\python test_webcam_vlm.py
```

---

## Lesson 6: Decoupled Multi-Modal Server (Llama.cpp & SmolVLM)

For production-grade robustness on consumer hardware, running Vision-Language Models (VLMs) directly inside PyTorch within the application process is sub-optimal. The modern standard is to decouple the inference engine entirely by running a local, lightweight C/C++ backend server (like `llama.cpp`'s `llama-server`) and querying it via a REST API.

### Advantages of the Decoupled Architecture:
1.  **Zero Python/PyTorch Overhead**: The VLM worker process doesn't import `torch` or `transformers`. This reduces process startup times from 15 seconds to 50ms and eliminates memory leaks.
2.  **Minimal VRAM Footprint**: A quantized GGUF model (like **SmolVLM-500M** or **SmolVLM2-2.2B**) runs natively in C++ using optimized CUDA kernels, consuming between **~700MB** and **~1.6GB** of VRAM (down from 2.5GB+ in PyTorch).
3.  **Numerical Stability**: The GGUF quantization formats (`Q8_0`, `Q4_K_M`) and C++ CUDA kernels are highly stable and completely immune to the PyTorch/bitsandbytes NaN-logits bug on Turing GTX cards (which caused text generators to spam exclamation marks `!`).
4.  **Error Isolation**: If the VLM server restarts or runs slow, the drone's telemetry and Flask server threads remain unaffected. The worker automatically handles connection drops and logs warnings instead of crashing the flight controller.

### Model Options
During the automated setup, you can choose between two model sizes:
*   **SmolVLM-500M-Instruct (Q8_0)**: The fastest and most lightweight option (~630MB total). Excellent for fast responses on low-end hardware.
*   **SmolVLM2-2.2B-Instruct (Q4_K_M)**: A larger, much more capable model (~1.5GB total). Recommended for better description quality and obstacle identification on 4GB VRAM cards like the GTX 1650.

### Setup and Startup Process
1.  **Run Setup**: `.\venv\Scripts\python setup_llama_cpp.py`
    *   This script queries the GitHub API for the latest Windows CUDA builds of `llama.cpp`.
    *   It downloads both the main binaries (`llama-server.exe`, etc.) and the CUDA runtime DLLs package, extracting them into the local `/llama_server/` folder.
    *   It prompts you to choose the model size and downloads the GGUF text model and mmproj visual projector into the `/llama_models/` folder.
    *   It generates a customized launch script (`run_llama_server.bat`).
2.  **Start VLM Server**: Run `.\run_llama_server.bat` in a dedicated terminal window and keep it open.
3.  **Start Autopilot Server**: Run `.\venv\Scripts\python -m drone.web_server` to connect the drone.
4.  **Test/Diagnose**: Run `.\venv\Scripts\python test_webcam_vlm.py` to check predictions locally with your PC webcam.