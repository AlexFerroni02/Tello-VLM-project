@echo off
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
llama-server.exe -m "..\llama_models\SmolVLM2-2.2B-Instruct-Q4_K_M.gguf" --mmproj "..\llama_models\mmproj-SmolVLM2-2.2B-Instruct-f16.gguf" -ngl 99 -c 2048 --port 8080

pause
