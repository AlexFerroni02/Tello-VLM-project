"""
VLM Autopilot Background Worker.
Contains the target loop for the secondary VLM processor process.
Communicates with a local llama.cpp server hosting SmolVLM, sending camera
frames as base64-encoded JPEG images via an HTTP REST API.
"""

import time
import sys
import os
import io
import base64
import requests
import numpy as np
from PIL import Image
from multiprocessing import Queue
from typing import Any, Optional

from drone.latency_queue import LatencyQueue
from config import VLM_SERVER_URL


def convert_frame_to_base64_jpeg(frame_rgb: np.ndarray) -> str:
    """
    Converts a NumPy RGB image frame into a base64-encoded JPEG string.

    Parameters:
        frame_rgb (np.ndarray): The input image frame as a NumPy array in RGB format.

    Returns:
        str: The base64-encoded JPEG image string, formatted as a data URL.

    Raises:
        ValueError: If the input array is empty or invalid.
    """
    if frame_rgb is None or frame_rgb.size == 0:
        raise ValueError("Input frame is empty or invalid.")

    # Convert NumPy array to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    
    # Save the PIL Image to an in-memory byte buffer as JPEG
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=80)
    
    # Encode binary JPEG data to Base64 bytes, decode to UTF-8 string
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Format as standard data URL expected by the llama.cpp server
    return f"data:image/jpeg;base64,{img_str}"


def vlm_worker_process(input_queue: LatencyQueue, output_queue: Queue, local_files_only: bool = False) -> None:
    """
    Main loop function executed by the secondary VLM worker process.
    Continuously pulls camera frames, sends them to the local llama-server
    API (SmolVLM), and posts the navigation analysis back to the dashboard.

    Parameters:
        input_queue (LatencyQueue): Capped IPC queue supplying camera frames.
        output_queue (Queue): Telemetry queue to return text co-pilot reports.
        local_files_only (bool): Unused parameter kept for API compatibility.

    Returns:
        None

    Raises:
        None
    """
    print(f"[VLM PROCESS] Background VLM worker process spawned successfully.")
    print(f"[VLM PROCESS] Target API URL: {VLM_SERVER_URL}")

    # Prompt sent to the VLM to instruct its environmental analysis
    flight_prompt = (
        "Describe what is in front of the camera in one short sentence, identifying obstacles and path safety."
    )

    # Initial check to wait for the local llama-server to become active
    server_online = False
    while not server_online:
        try:
            # Test connection to the server's completions or health endpoint
            # We use a short timeout to prevent blocking indefinitely
            test_resp = requests.get(VLM_SERVER_URL.replace("/v1/chat/completions", "/health"), timeout=2.0)
            if test_resp.status_code in (200, 503):  # 503 means model is loading but server is up
                server_online = True
                print("[VLM PROCESS] Llama-server detected online!")
            else:
                print(f"[VLM PROCESS] Server returned status code {test_resp.status_code}. Waiting for initialization...")
                time.sleep(2.0)
        except requests.exceptions.RequestException:
            print("[VLM PROCESS] Llama-server is offline. Please launch run_llama_server.bat. Retrying in 2s...")
            output_queue.put("Offline. Waiting for llama.cpp server...")
            time.sleep(2.0)

    output_queue.put("VLM autopilot server connected. Running visual safety scans...")

    try:
        while True:
            # Read frame from the LatencyQueue (RGB format).
            # This blocks until a frame is published by the main server process.
            frame_rgb = input_queue.get()

            # Poison pill check: if the main process sends None, terminate cleanly
            if frame_rgb is None:
                break

            start_time = time.time()

            try:
                # 1. Convert frame to base64 JPEG format
                base64_image = convert_frame_to_base64_jpeg(frame_rgb)

                # 2. Build the OpenAI-compatible chat completion payload
                payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": flight_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": base64_image
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.2,
                    "max_tokens": 128
                }

                # 3. Post HTTP request to local llama-server
                headers = {"Content-Type": "application/json"}
                response = requests.post(VLM_SERVER_URL, json=payload, headers=headers, timeout=60.0)

                # 4. Process response
                if response.status_code == 200:
                    result_data = response.json()
                    answer = result_data["choices"][0]["message"]["content"]
                    answer_clean = answer.strip().replace("\n", " ")

                    # Calculate latency for logging purposes
                    elapsed = time.time() - start_time
                    print(f"[VLM PROCESS] Scan complete in {elapsed:.2f}s -> {answer_clean}")

                    # Return the result to the main thread
                    output_queue.put(answer_clean)
                else:
                    error_msg = f"HTTP Error {response.status_code}: {response.text}"
                    print(f"[VLM PROCESS] Server Error: {error_msg}")
                    output_queue.put(f"VLM server error: {error_msg}")

            except requests.exceptions.RequestException as req_err:
                print(f"[VLM PROCESS] Request failed: {req_err}")
                output_queue.put("Connection lost to llama.cpp server. Retrying...")
                time.sleep(1.0)
            except Exception as inf_err:
                import traceback
                print("[VLM PROCESS] Inference exception traceback:")
                traceback.print_exc()
                output_queue.put(f"VLM scan error: {inf_err}")

    except KeyboardInterrupt:
        print("[VLM PROCESS] Worker loop interrupted by keyboard signal.")
    except Exception as loop_err:
        print(f"[VLM PROCESS] Error in background loop: {loop_err}")

    print("[VLM PROCESS] Background VLM worker process shut down cleanly.")
