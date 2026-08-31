#!/bin/bash
# install.sh - Raspberry Pi Virtual Light setup
# Tested on Raspberry Pi OS Bookworm (64-bit) - Pi 4 / Pi 5
set -e
echo "=== Virtual Light - Pi Installer ==="

if [[ $EUID -eq 0 ]]; then
  echo "Do not run as root (sudo will be used when needed)"
  exit 1
fi

echo "[1/5] Updating system..."
sudo apt update

echo "[2/5] Installing system deps (OpenCV, V4L2, libcamera)..."
sudo apt install -y \
  python3-pip python3-venv python3-picamera2 \
  libcamera-apps \
  libatlas-base-dev \
  libhdf5-dev libopenjp2-7 \
  python3-opencv \
  v4l-utils

echo "[3/5] Creating venv..."
python3 -m venv venv --system-site-packages
# --system-site-packages allows picamera2 + system opencv reuse (faster, less memory)
source venv/bin/activate

echo "[4/5] Installing Python deps..."
pip install --upgrade pip
# Use piwheels for ARM wheels (much faster)
pip install --extra-index-url https://www.piwheels.org/simple \
  opencv-python==4.8.1.78 \
  numpy==1.26.4 \
  mediapipe==0.10.14

# Optional: if you want TFLite instead of MediaPipe, uncomment:
# pip install tflite-runtime==2.14.0

echo "[5/5] Performance tuning..."
# Enable hardware optimizations
echo "Applying Pi performance tweaks (requires reboot to fully apply):"

# Increase GPU mem for camera (Pi 4)
if grep -q "gpu_mem" /boot/firmware/config.txt 2>/dev/null; then
  echo "gpu_mem already set"
else
  echo "gpu_mem=128" | sudo tee -a /boot/firmware/config.txt
fi

# Enable 64-bit + arm_boost for Pi 4
sudo raspi-config nonint do_memory_split 128 || true

echo ""
echo "=== Install complete ==="
echo "Activate: source venv/bin/activate"
echo "Run (Pi5): python main.py --profile pi5"
echo "Run (Pi4): python main.py --profile pi4"
echo "Run (Low): python main.py --profile pi4_low --backend color"
echo "Run (PC) : python main.py --profile pc_debug"
echo ""
echo "Tips:"
echo "  - If MediaPipe is slow, use --backend color (HSV glove/marker)"
echo "  - Calibrate HSV: python tools/calibrate_color.py"
echo "  - Benchmark: python benchmark.py"
