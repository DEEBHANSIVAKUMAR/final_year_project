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

- MediaPipe Face Mesh landmark tracking
- Head-pose estimation using OpenCV `solvePnP`
- Automatic centre-position calibration
- Stable command confirmation across multiple frames
- Fail-safe `STOP` when a face is missing, calibration is incomplete, or tracking is unstable
- Raspberry Pi Camera / USB webcam support
- Pi 5, Pi 4, low-end Pi, and PC-debug profiles
- On-screen FPS, latency, pose, and wheelchair-command display

## How It Works

```text
Camera frame
    → MediaPipe Face Mesh
    → Head-pose estimation (yaw / pitch)
    → Centre calibration + smoothing
    → Command confirmation
    → LEFT / RIGHT / FORWARD / BACKWARD / STOP
```

At startup, the user keeps their face straight while the system records a neutral head pose. Future pose angles are measured relative to this calibrated centre. The default patient-control configuration deliberately requires a clear, sustained head turn before it shows a movement command. Small movement, an unstable pose, or lost face tracking immediately results in `STOP`.

## Requirements

- Raspberry Pi 4 or Raspberry Pi 5 recommended
- Raspberry Pi Camera Module or USB webcam
- Raspberry Pi OS Bookworm (64-bit) recommended
- Python 3.11+ on the Raspberry Pi

Python packages are listed in [requirements.txt](requirements.txt):

- `opencv-python`
- `numpy`
- `mediapipe==0.10.14`

## Installation on Raspberry Pi

Clone or copy this project to the Pi, then run:

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

## Configuration

Main direction settings are in [config.py](config.py):

```python
HEAD_YAW_THRESHOLD = 30.0      # clear LEFT / RIGHT head turn required
HEAD_PITCH_THRESHOLD = 22.0    # clear FORWARD / BACKWARD head tilt required
COMMAND_CONFIRM_FRAMES = 15    # sustained frames required before movement command
HEAD_DIRECTION_INVERT_X = False
HEAD_DIRECTION_INVERT_Y = False
```

- Increase a threshold if accidental commands occur.
- Decrease a threshold only after careful controlled testing.
- Increase `COMMAND_CONFIRM_FRAMES` for a more conservative response.
- Set an inversion value to `True` if the corresponding direction is reversed on the installed camera.

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
