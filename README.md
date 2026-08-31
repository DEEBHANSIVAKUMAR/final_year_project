# Real-time Virtual Light System — Raspberry Pi Optimized

> **Priority: Real-Time Performance → Low Latency → Stable Tracking → Visual Quality**
> Targets 20–30 FPS on Pi 4 / 30 FPS on Pi 5 with <50ms latency.

---

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
chmod +x install.sh
./install.sh
source venv/bin/activate
python main.py --profile pi5      # Pi 5
# or
python main.py --profile pi4      # Pi 4
python main.py --profile pi4_low --backend color  # Pi 4 1GB / Pi 3B+ / Zero 2 W
```

### Manual install
```bash
sudo apt update
sudo apt install -y python3-pip python3-venv python3-picamera2 libcamera-apps python3-opencv v4l-utils

python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip
pip install --extra-index-url https://www.piwheels.org/simple opencv-python numpy mediapipe==0.10.14
```

> `mediapipe==0.10.14` has piwheels ARM wheel; no build needed. If you use Pi Camera Module 3, `picamera2` must come from apt (not pip).

---

## Usage

```bash
python main.py                          # defaults: pi5 + auto backend
python main.py --profile pi4 --backend color
python main.py --profile pc_debug       # on laptop/PC

# Keys while running
q / ESC  quit
c        cycle light color
m        toggle mediapipe ↔ color tracker
s        save snapshot
```

**Tune HSV color:** `python tools/calibrate_color.py` → copy `HSV_LOWER/UPPER` into `config.py`.

**Benchmark without camera:** `python benchmark.py --profile pi4 --backend color` (also tests mediapipe if installed).

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | 4.8+ | Capture, resize, blend, display (NEON accel) |
| `numpy` | 1.24+ | Vectorized glow texture + smoothing |
| `mediapipe` | 0.10.14 | Hands Lite detector (optional, fallback to color) |
| `picamera2` | apt | Hardware-accelerated Pi camera (optional) |
| `tflite-runtime` | 2.14 (optional) | Alternative if MediaPipe not available |

Full list: `requirements.txt`

---

## Performance Optimization Techniques (applied)

| # | Technique | Saving |
|---|---|---|
| 1 | Threaded capture + queue size 1 | ~30ms latency |
| 2 | MJPG FourCC + `BUFFERSIZE=1` | 3× vs YUYV |
| 3 | Picamera2 zero-copy on Pi 5 | 5–8ms |
| 4 | Detect at 256×192 (6.2× fewer pixels) | ~6× speedup |
| 5 | Frame skipping (detect every 2–3 frames) | 50–66% detector load |
| 6 | Precomputed glow (blur once at init) | ~8ms/frame saved |
| 7 | ROI-only blend (in-place, no full-frame copy) | 10× memory BW |
| 8 | `cv2.setUseOptimized(True)` + `setNumThreads(2)` | 15–20% FPS |
| 9 | Adaptive One-Euro smoother | jitter ↓ without lag |
| 10 | RGB convert only for MediaPipe | saves 1 cvt per skipped frame |

---

## Camera Resolution & FPS Recommendations

| Hardware | Capture | Detect | Skip | Expect | Glow radius |
|---|---|---|---|---|---|
| Pi 5 8GB | 640×480 | 320×240 | 1 | 28–30 FPS | 80 |
| Pi 4 4/8GB | 640×480 | 256×192 | 2 | 24–28 FPS | 60 |
| Pi 4 2GB | 480×360 | 256×192 | 2 | 24 FPS | 60 |
| Pi 4 1GB / 3B+ | 480×360 | 192×144 | 3 | 20 FPS | 50 (color backend) |
| Zero 2 W | 480×360 | 192×144 | 3 | 15–20 FPS | 50 color only |

> Never use 1280×720 for this effect — no visual gain, 2.25× cost.

---

## Latency Reduction Checklist

- MJPG, `BUFFERSIZE=1`, threaded queue=1, small detect res, skip frames, precomputed texture, `waitKey(1)`, 64-bit OS, `arm_boost=1` (Pi 4), active cooling.

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

```
final_year_project/
├── config.py            # profiles + tunables
├── tracker.py           # MediaPipe + Color trackers + smoother
├── virtual_light.py     # precomputed glow texture + blend
├── main.py              # threaded camera + main loop + metrics
├── benchmark.py         # synthetic benchmark (no camera)
├── tools/calibrate_color.py  # HSV slider tuner
├── requirements.txt
├── install.sh
├── ARCHITECTURE.md      # full tech comparison + diagram
└── README.md
```

---

## License

MIT — use for final year project, cite MediaPipe & OpenCV.
