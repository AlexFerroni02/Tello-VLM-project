"""
Tello Web Autopilot Server.
Serves a Flask web application that hosts a local dashboard.
Exposes REST endpoints for telemetry, video streaming (MJPEG), 
and non-blocking RC flight controls.
"""

import os
import time
import sys
import cv2
import multiprocessing
from multiprocessing import Queue
from typing import Generator
from flask import Flask, render_template, Response, jsonify, request
from drone.tello_drone import TelloDrone
from drone.latency_queue import LatencyQueue
from drone.vlm_worker import vlm_worker_process

# Initialize Flask. We explicitly configure the template folder 
# to look inside the project root's 'templates' directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "../templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Initialize the global TelloDrone reference (will be populated inside start_server)
# to avoid WinError 10048 socket conflicts during multiprocessing spawn on Windows.
drone = None

# Global multiprocessing communication channels
vlm_input_queue = None
vlm_output_queue = None
latest_vlm_analysis = "Waiting for initial VLM scan..."
vlm_prompt = "Describe what is in front of the camera in one short sentence, identifying obstacles and path safety."

def generate_video_stream(drone_instance: TelloDrone) -> Generator[bytes, None, None]:
    """
    Generator function that yields MJPEG video chunks from the drone camera.
    Converts RGB frames to BGR, encodes them as JPEG, and wraps them in boundary bytes.
    
    Parameters:
        drone_instance (TelloDrone): The active drone controller wrapper.
        
    Yields:
        bytes: MJPEG frame chunk in bytes format.
    """
    print("Web Video generator thread started.")
    while True:
        frame_rgb = drone_instance.get_frame()
        if frame_rgb is None:
            # Stream not ready, wait a bit and retry (approx 30 FPS boundary)
            time.sleep(0.03)
            continue
            
        # Extract downscaled VLM frame (378x378) and push it to the VLM queue
        # ONLY if the queue is empty (meaning the VLM background process is ready).
        # This reduces pickling overhead from 30 FPS to ~1.2 FPS, freeing CPU and GIL.
        vlm_frame = drone_instance.get_vlm_frame()
        if vlm_frame is not None and vlm_input_queue is not None:
            if vlm_input_queue.empty():
                # Push a packet containing both the frame and the dynamic VLM prompt
                vlm_input_queue.put_latest({
                    "frame": vlm_frame,
                    "prompt": vlm_prompt
                })
            
        # Convert RGB to BGR for JPEG encoder
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Encode the frame as JPEG
        success, buffer = cv2.imencode('.jpg', frame_bgr)
        if not success:
            continue
            
        # Format the frame into multipart/x-mixed-replace chunk bytes
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Throttle loop to prevent overwhelming CPU (~33ms intervals)
        time.sleep(0.03)

@app.route('/')
def index():
    """
    Renders the central Web Cockpit Autopilot interface.
    
    Returns:
        HTML: Renders the templates/index.html page.
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    MJPEG Video stream endpoint. The browser native <img> tag reads from this.
    
    Returns:
        Response: Flask response stream yielding MJPEG frames.
    """
    return Response(
        generate_video_stream(drone),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/telemetry')
def telemetry():
    """
    REST API endpoint to poll the latest drone sensor data.
    
    Returns:
        JSON: Current telemetry metrics cache.
    """
    data = drone.get_telemetry()
    return jsonify(data)

@app.route('/control', methods=['POST'])
def control():
    """
    REST API endpoint to set active non-blocking RC flight velocities.
    Expects JSON data with 'roll', 'pitch', 'throttle', and 'yaw' fields.
    
    Returns:
        JSON: Connection and command status.
    """
    if not drone.is_connected:
        return jsonify({"status": "error", "message": "Drone not connected"}), 400
        
    data = request.json or {}
    roll = int(data.get('roll', 0))
    pitch = int(data.get('pitch', 0))
    throttle = int(data.get('throttle', 0))
    yaw = int(data.get('yaw', 0))
    
    # Send continuous velocity commands (non-blocking)
    # Tello accepts values between -100 and +100
    drone.tello.send_rc_control(roll, pitch, throttle, yaw)
    return jsonify({"status": "ok"})

@app.route('/takeoff', methods=['POST'])
def takeoff():
    """
    REST API endpoint to trigger automatic drone takeoff.
    
    Returns:
        JSON: Takeoff execution result.
    """
    success = drone.takeoff()
    if success:
        return jsonify({"status": "ok", "message": "Takeoff completed"})
    else:
        return jsonify({"status": "error", "message": "Takeoff failed"}), 500

@app.route('/land', methods=['POST'])
def land():
    """
    REST API endpoint to trigger automatic drone landing.
    
    Returns:
        JSON: Landing execution result.
    """
    success = drone.land()
    if success:
        return jsonify({"status": "ok", "message": "Landing completed"})
    else:
        return jsonify({"status": "error", "message": "Landing failed"}), 500

@app.route('/shutdown', methods=['POST'])
def shutdown():
    """
    REST API endpoint to trigger full system shutdown.
    Commands the drone to land, terminates background threads/processes, 
    and kills the Flask server process.
    
    Returns:
        JSON: Shutdown acknowledgment.
    """
    print("SYSTEM SHUTDOWN requested from Web Console. Shutting down in 1 second...")
    
    import threading
    import signal
    
    def kill_server_process():
        time.sleep(1.0)
        # Send SIGINT to our own process PID.
        # This will trigger the 'finally' block in start_server() on Windows.
        os.kill(os.getpid(), signal.SIGINT)
        
    threading.Thread(target=kill_server_process).start()
    return jsonify({"status": "ok", "message": "System shutdown initiated. Landing drone..."})

@app.route('/vlm_log')
def vlm_log():
    """
    REST API endpoint to retrieve the latest cached VLM analysis text.
    Checks for any new responses in the vlm_output_queue without blocking.
    
    Returns:
        JSON: Contains the latest text description of the environment.
    """
    global latest_vlm_analysis
    if vlm_output_queue is not None:
        try:
            # Drain the queue to fetch the most recent prediction
            while not vlm_output_queue.empty():
                latest_vlm_analysis = vlm_output_queue.get_nowait()
        except Exception:
            pass
    return jsonify({"log": latest_vlm_analysis})

@app.route('/update_prompt', methods=['POST'])
def update_prompt():
    """
    REST API endpoint to dynamically update the active flight prompt sent to the VLM.
    """
    global vlm_prompt
    data = request.json or {}
    new_prompt = data.get('prompt', '').strip()
    if new_prompt:
        vlm_prompt = new_prompt
        print(f"[SYSTEM] Active VLM flight prompt updated to: {vlm_prompt}")
        return jsonify({"status": "ok", "prompt": vlm_prompt})
    return jsonify({"status": "error", "message": "Invalid prompt string"}), 400

def start_server():
    """
    Initializes the Tello connection, starts the video capture stream,
    spawns the background VLM worker process, and runs the Flask server on port 5000.
    Ensures safe shutdown and landing inside a finally block if server exits.
    
    Parameters:
        None
        
    Returns:
        None
    """
    global drone, vlm_input_queue, vlm_output_queue
    
    drone = TelloDrone()
    print("Connecting to Tello...")
    if not drone.connect():
        print("\n" + "="*80)
        print("[WARNING] Could not establish connection to Tello at 192.168.10.1.")
        print("[WARNING] Entering DEMO/MOCK mode. Web UI will load, and VLM background worker")
        print("[WARNING] will initialize and download/cache models, but drone commands will be simulated.")
        print("="*80 + "\n")
    else:
        print("Starting Tello video feed reader...")
        if not drone.start_video():
            print("[WARNING] Failed to initialize camera feed. Video streaming will be disabled.")
        
    # Initialize process-safe queues
    vlm_input_queue = LatencyQueue()
    vlm_output_queue = Queue()
    
    # Determine offline model loading mode based on Tello connectivity status.
    # If the drone is connected, we are flying offline on Tello's Wi-Fi (local_files_only = True).
    # If the drone is disconnected, we are on Home Wi-Fi (local_files_only = False for downloading).
    local_files_only = drone.is_connected

    # Spawn the background VLM worker process
    print(f"Spawning VLM background process (Offline Mode: {local_files_only})...")
    vlm_process = multiprocessing.Process(
        target=vlm_worker_process,
        args=(vlm_input_queue, vlm_output_queue, local_files_only),
        daemon=True
    )
    vlm_process.start()
    print(f"VLM worker process spawned with PID: {vlm_process.pid}")
    
    try:
        # Run local server on host '0.0.0.0' to allow connections from local PC
        # We disable Flask's debug mode reloader because it spawns duplicate subprocesses,
        # which would try to connect to the Tello UDP socket twice and trigger errors.
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
        
    finally:
        print("\nWeb Server shutdown detected. Initiating flight safety locks...")
        
        # Clean terminate of the VLM worker process
        if 'vlm_process' in locals() and vlm_process.is_alive():
            print("Terminating VLM worker process...")
            vlm_process.terminate()
            vlm_process.join(timeout=2.0)
            print("VLM worker process terminated.")
            
        if drone.is_connected:
            try:
                print("Safety Check: Commanding drone landing before socket release...")
                drone.land()
            except Exception as safety_err:
                print(f"Safety landing failed: {safety_err}")
        
        drone.disconnect()
        print("Server shutdown completed safely.")

if __name__ == "__main__":
    start_server()
