"""
Webcam VLM Diagnostic Script (Llama.cpp & SmolVLM Edition).
Captures frames from the PC webcam using OpenCV and runs visual inference
by querying the local llama-server API (http://localhost:8080).
"""

import cv2
import base64
import requests
import io
from PIL import Image

VLM_SERVER_URL = "http://localhost:8080/v1/chat/completions"


def main() -> None:
    """
    Captures frames from the PC webcam and sends them to the local llama-server
    for text analysis when SPACEBAR is pressed.
    """
    print("="*60)
    print("WEBCAM VLM TEST INTERFACE (LLAMA.CPP)")
    print("- Press SPACEBAR to capture the frame and run VLM analysis.")
    print("- Press 'q' or 'ESC' to exit.")
    print("="*60 + "\n")

    # Connect to the local server
    try:
        resp = requests.get(VLM_SERVER_URL.replace("/v1/chat/completions", "/health"), timeout=2.0)
        if resp.status_code in (200, 503):
            print("Connected to llama-server successfully!")
        else:
            print(f"Server health returned status code: {resp.status_code}")
    except requests.exceptions.RequestException:
        print("CRITICAL ERROR: Could not connect to llama-server.")
        print("Please start the server first by running .\\run_llama_server.bat in another terminal.")
        return

    # Initialize PC webcam (index 0 is typically the built-in webcam)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Display the frame to the user
        cv2.imshow("Webcam VLM Test (Press SPACE to Analyze, Q to Quit)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            print("\n[VLM] Analyzing captured frame...")
            
            # Convert BGR OpenCV frame to PIL RGB image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # Convert PIL image to base64 JPEG format
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=80)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            base64_image = f"data:image/jpeg;base64,{img_str}"

            # Query payload
            payload = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe what you see in front of you in one short sentence."
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
                "max_tokens": 128
            }

            try:
                # Post HTTP request to local llama-server
                headers = {"Content-Type": "application/json"}
                response = requests.post(VLM_SERVER_URL, json=payload, headers=headers, timeout=60.0)

                if response.status_code == 200:
                    answer = response.json()["choices"][0]["message"]["content"]
                    print(f"[VLM ANSWER] {answer.strip()}\n")
                else:
                    print(f"[VLM ERROR] HTTP Error {response.status_code}: {response.text}\n")
            except Exception as e:
                print(f"[VLM ERROR] Failed to contact server: {e}\n")

        elif key == ord('q') or key == 27:  # 'q' or ESC
            print("Exiting...")
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
