# Smart Wheelchair Direction Detection

A Raspberry Pi–based assistive-technology prototype that detects head movement through a camera and produces stable navigation commands for a smart wheelchair.

The system uses **MediaPipe Face Mesh** facial landmarks and OpenCV head-pose estimation rather than face-box movement. It detects the user's head orientation and displays one of five commands:

| Head movement | Display command |
|---|---|
| Turn left | `LEFT` |
| Turn right | `RIGHT` |
| Look up | `FORWARD` |
| Look down | `BACKWARD` |
| Face straight, no face, or unstable pose | `STOP` |

> **Safety notice:** This repository currently displays direction commands only. It does **not** control motors or GPIO pins. Do not connect camera output directly to a wheelchair motor controller. A physical emergency-stop, manual control override, speed limits, obstacle detection, and hardware testing are required before any real mobility use.

## Features
- Threaded camera (Picamera2 + V4L2 MJPG) with queue=1 → minimal lag
- Dual tracker: **MediaPipe Hands Lite** (primary) + **HSV Color** (fallback, 3ms)
- Downscaled detection (256×192 / 320×240) + frame skipping
- Precomputed radial glow texture → per-frame cost ~0.8ms (no blur per frame)
- One-Euro adaptive smoothing → no jitter, no lag
- FPS + latency overlay, hot-switch `c`olor / `m`ediapipe, snapshot `s`
- Profiles for Pi 5 / Pi 4 / Pi 4 Low (1GB) / PC debug

---

## Architecture at a Glance

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for full comparison (OpenCV vs MediaPipe vs TFLite vs YOLO) and performance budget.

**Selected:** MediaPipe Hands Lite (`complexity=0`) for fingertip accuracy + HSV color fallback for ultra-low-end. Full diagram + latency budget in ARCHITECTURE.md:5.

```
Camera (640×480 MJPG, queue=1) → Resize 256×192 → MediaPipe Lite / HSV → Scale + Smooth → Precomputed Glow ROI Blend → Display
```

---

## Installation — Raspberry Pi OS Bookworm (64-bit)

### Quick start
```bash
cd /path/to/final_year_project
chmod +x install.sh
./install.sh
source venv/bin/activate
```

The installer installs Raspberry Pi camera tools, OpenCV, NumPy, and MediaPipe.

## Run the Project

### Raspberry Pi 5 with Pi Camera

```bash
source venv/bin/activate
python3 main.py --profile pi5 --backend face
```

### Raspberry Pi 4

```bash
source venv/bin/activate
python3 main.py --profile pi4 --backend face
```

### USB Webcam

```bash
source venv/bin/activate
python3 main.py --profile pi5 --backend face --no-picamera2
```

### Low-end Pi profile

```bash
source venv/bin/activate
python3 main.py --profile pi4_low --backend face --no-picamera2
```

## Controls

| Key | Action |
|---|---|
| `r` | Reset calibration; keep the face straight again |
| `q` or `Esc` | Quit |
| `m` | Cycle tracking backends for testing |
| `f` | Toggle face / hand tracking demo mode |
| `s` | Save a camera snapshot |

## Calibration and Testing

1. Mount the camera securely in front of the user at face height.
2. Start the application and face the camera straight for approximately 30 frames.
3. Confirm that the display changes from `CALIBRATING` to `STOP`.
4. Turn the head slowly in each direction and check the shown command.
5. Press `r` whenever the seating position or camera position changes.
6. If left/right or forward/backward appears reversed, adjust the inversion options in `config.py`.

Test the commands on screen first. Do not use the system with a person seated in a powered wheelchair until a separately tested and supervised motor-control safety system is implemented.

## Light Rendering — Lightweight

Radial gradient `falloff=(1 - dist/r)^1.8` blurred once with `σ=0.35·r`. Per frame: ROI `cv2.add` (NEON). Inner core = 2 circles. No per-frame blur, no shaders. See `virtual_light.py:14`.

---

## Pi 4 vs Pi 5 Notes

- **Pi 5** (A76 2.4GHz, 4267 MT/s RAM): MediaPipe Lite ~8–15ms, can do 320×240 every frame.
- **Pi 4** (A72 1.5GHz): 12–22ms → must skip frames, use 256×192, active cooling to avoid throttling (80 °C). Overclock to 1.8GHz if stable (`arm_freq=1800` in `/boot/firmware/config.txt`).

---

## Troubleshooting

- `Cannot open camera` → check `v4l2-ctl --list-devices`, try `--camera 1`, ensure `libcamera-hello --timeout 1000` works for Pi camera.
- Low FPS (<15) → `python benchmark.py`, then switch to `--backend color` or `--profile pi4_low`.
- Jitter → lower `SMOOTHING_ALPHA` in `config.py` (0.4 = smoother, 0.8 = responsive).
- Mediapipe not found → auto falls back to color tracker (needs green marker).

---

## Project Structure

```text
final_year_project/
├── main.py                    # Camera loop, command display, keyboard controls
├── tracker.py                 # Face Mesh, head pose, calibration, command confirmation
├── config.py                  # Hardware profiles and detection thresholds
├── virtual_light.py           # Earlier visual-light demo component
├── benchmark.py               # Performance benchmark utility
├── tools/calibrate_color.py   # HSV colour-tracker calibration utility
├── requirements.txt
├── install.sh
└── ARCHITECTURE.md             # Earlier virtual-light architecture notes
```

## Future Work

- Add an isolated motor-controller interface after hardware selection
- Add physical emergency-stop and manual joystick override
- Add obstacle detection with ultrasonic, LiDAR, or depth camera sensors
- Add command timeout and maximum-speed enforcement at the motor-controller layer
- Record test data to tune thresholds for individual users

## License

MIT License. This is an academic prototype and is not a certified medical or mobility device.
