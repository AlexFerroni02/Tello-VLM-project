"""
Llama.cpp & SmolVLM Automated Setup Script.
Queries GitHub API for the latest llama.cpp Windows CUDA 12 binaries, downloads them,
downloads the SmolVLM GGUF models from Hugging Face, and creates a startup batch file.
"""

import os
import sys
import json
import zipfile
import urllib.request
import ssl

# Directory configurations
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_SERVER_DIR = os.path.join(PROJECT_DIR, "llama_server")
LLAMA_MODELS_DIR = os.path.join(PROJECT_DIR, "llama_models")

# SmolVLM model options
MODEL_OPTIONS = {
    "1": {
        "name": "SmolVLM-500M-Instruct (Q8_0)",
        "model_url": "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/SmolVLM-500M-Instruct-Q8_0.gguf",
        "mmproj_url": "https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-500M-Instruct-f16.gguf",
        "model_file": "SmolVLM-500M-Instruct-Q8_0.gguf",
        "mmproj_file": "mmproj-SmolVLM-500M-Instruct-f16.gguf"
    },
    "2": {
        "name": "SmolVLM2-2.2B-Instruct (Q4_K_M)",
        "model_url": "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        "mmproj_url": "https://huggingface.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF/resolve/main/mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
        "model_file": "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        "mmproj_file": "mmproj-SmolVLM2-2.2B-Instruct-f16.gguf"
    }
}


def download_with_progress(url: str, dest_path: str, description: str) -> None:
    """
    Downloads a file showing progress in the console.
    """
    print(f"Downloading {description}...")
    
    # Bypass SSL verification if needed
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as response:
            total_size = int(response.info().get('Content-Length', 0))
            downloaded = 0
            block_size = 8192
            
            with open(dest_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    if total_size > 0:
                        percent = min(100, int(downloaded * 100 / total_size))
                        sys.stdout.write(f"\rProgress: {percent}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)")
                    else:
                        sys.stdout.write(f"\rProgress: {downloaded / 1024 / 1024:.1f}MB downloaded")
                    sys.stdout.flush()
        print("\nDownload complete!\n")
    except Exception as e:
        print(f"\nError downloading: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        sys.exit(1)


def main() -> None:
    """
    Downloads and extracts llama.cpp Windows CUDA 12 binaries, downloads SmolVLM GGUF,
    and sets up startup files.
    """
    # Create directories
    os.makedirs(LLAMA_SERVER_DIR, exist_ok=True)
    os.makedirs(LLAMA_MODELS_DIR, exist_ok=True)

    print("="*60)
    print("AUTOMATED LLAMA.CPP & SMOLVLM SETUP")
    print("="*60)

    # Prompt user for model size
    print("Select the SmolVLM model size to download and configure:")
    print("1) SmolVLM-500M-Instruct (Q8_0, ~630MB total, fast and lightweight)")
    print("2) SmolVLM2-2.2B-Instruct (Q4_K_M, ~1.5GB total, more descriptive, recommended for 4GB VRAM)")
    
    selected_option = "2" # Default to 2.2B since user requested it
    try:
        user_choice = input("Enter option (1 or 2, default is 2): ").strip()
        if user_choice in ("1", "2"):
            selected_option = user_choice
    except Exception:
        # Fallback if running in non-interactive environment
        pass
        
    model_cfg = MODEL_OPTIONS[selected_option]
    print(f"\nConfigured to download: {model_cfg['name']}\n")



    # --- Step 1: Find latest llama.cpp CUDA 12 releases (Executables & DLLs) ---
    print("Querying GitHub API for the latest llama.cpp release...")
    
    # Custom headers to bypass GitHub API rate limits/errors
    req = urllib.request.Request(
        "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest",
        headers={"User-Agent": "Mozilla/5.0"}
    )
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            release_info = json.loads(response.read().decode())
    except Exception as e:
        print(f"Failed to query GitHub API: {e}")
        sys.exit(1)

    print(f"Latest release found: {release_info['tag_name']}")
    
    # Search for main Windows CUDA 12 asset (excluding cudart dll wrapper)
    main_asset = None
    for asset in release_info.get("assets", []):
        name = asset["name"].lower()
        if "bin-win" in name and "x64.zip" in name and ("cuda" in name or "cu" in name) and "cudart" not in name:
            main_asset = asset
            break

    # Fallback to any win-x64 zip without cudart
    if not main_asset:
        for asset in release_info.get("assets", []):
            name = asset["name"].lower()
            if "win" in name and "x64.zip" in name and "cudart" not in name:
                main_asset = asset
                break

    # Search for matching cudart CUDA runtime DLLs
    cudart_asset = None
    for asset in release_info.get("assets", []):
        name = asset["name"].lower()
        if "cudart" in name and "win" in name and "cuda" in name and "x64.zip" in name:
            cudart_asset = asset
            break

    if not main_asset:
        print("Error: Could not find Windows binary package in assets.")
        print("Available assets were:")
        for asset in release_info.get("assets", []):
            print(f"- {asset['name']}")
        sys.exit(1)

    # --- Step 2: Download & Extract BOTH packages ---
    temp_zip = os.path.join(PROJECT_DIR, "llama_temp.zip")

    # Download and extract main binaries
    print(f"Selected Main Binary Asset: {main_asset['name']}")
    download_with_progress(main_asset["browser_download_url"], temp_zip, "llama.cpp executables")
    print("Extracting llama.cpp executables...")
    try:
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(LLAMA_SERVER_DIR)
        print("Executables extracted successfully!\n")
    except Exception as e:
        print(f"Failed to extract executables: {e}")
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
        sys.exit(1)
    if os.path.exists(temp_zip):
        os.remove(temp_zip)

    # Download and extract CUDA runtime DLLs if available
    if cudart_asset:
        print(f"Selected CUDA DLLs Asset: {cudart_asset['name']}")
        download_with_progress(cudart_asset["browser_download_url"], temp_zip, "CUDA runtime DLLs")
        print("Extracting CUDA runtime DLLs...")
        try:
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(LLAMA_SERVER_DIR)
            print("CUDA DLLs extracted successfully!\n")
        except Exception as e:
            print(f"Failed to extract CUDA DLLs: {e}")
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
            sys.exit(1)
        if os.path.exists(temp_zip):
            os.remove(temp_zip)
    else:
        print("No matching CUDA runtime DLLs package found in assets. Assuming they are pre-bundled or CUDA Toolkit is globally installed.")

    # --- Step 3: Download SmolVLM GGUF models ---
    # Download main GGUF model
    model_dest = os.path.join(LLAMA_MODELS_DIR, model_cfg["model_file"])
    if os.path.exists(model_dest):
        print(f"{model_cfg['model_file']} already exists. Skipping download.")
    else:
        download_with_progress(model_cfg["model_url"], model_dest, f"SmolVLM Text Model ({model_cfg['model_file']})")

    # Download visual mmproj GGUF model
    mmproj_dest = os.path.join(LLAMA_MODELS_DIR, model_cfg["mmproj_file"])
    if os.path.exists(mmproj_dest):
        print(f"{model_cfg['mmproj_file']} already exists. Skipping download.")
    else:
        download_with_progress(model_cfg["mmproj_url"], mmproj_dest, f"SmolVLM mmproj visual projector ({model_cfg['mmproj_file']})")

    # --- Step 4: Create run_llama_server.bat batch script ---
    bat_path = os.path.join(PROJECT_DIR, "run_llama_server.bat")
    print(f"Creating startup script: {bat_path}...")
    
    bat_content = f"""@echo off
title Llama.cpp SmolVLM Server
echo Starting llama-server with SmolVLM model (CUDA Acceleration Enabled)...
cd /d "%~dp0llama_server"

REM Check if llama-server.exe exists
if not exist llama-server.exe (
    echo Error: llama-server.exe not found in llama_server folder!
    pause
    exit /b 1
)

REM Launch llama-server with GPU offload (ngl 99) and 2048 context length on port 8080
llama-server.exe -m "..\\llama_models\\{model_cfg['model_file']}" --mmproj "..\\llama_models\\{model_cfg['mmproj_file']}" -ngl 99 -c 2048 --port 8080

pause
"""
    try:
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write(bat_content)
        print("Startup script created successfully!\n")
    except Exception as e:
        print(f"Failed to create startup script: {e}")

    print("="*60)
    print("SETUP COMPLETE!")
    print("You can now start the VLM server by running:")
    print("  .\\run_llama_server.bat")
    print("="*60)


if __name__ == "__main__":
    main()
