# Virtual Light System — Architecture & Technology Comparison

## 1. Goal
Real-time virtual light that follows fingertip/hand at 20–30 FPS on Raspberry Pi with <50ms end-to-end latency.

## 2. Approach Comparison

| Approach | Model Size | Pi 4 Latency (320x240) | Pi 5 Latency | Pros | Cons | Verdict |
|---|---|---|---|---|---|---|
| **OpenCV Classical (HSV + Contour)** | 0 MB | **2–4 ms** | 1–2 ms | No DL, deterministic, 30 FPS+ on Pi Zero | Needs colored marker/glove, lighting sensitive | **Fallback / ultra-low-end** |
| **MediaPipe Hands Lite (complexity=0)** | ~12 MB (tflite) | **12–22 ms** | 8–15 ms | No training, robust hand landmark (21 pts), index fingertip stable | 1 hand only at high FPS, needs good light | **RECOMMENDED for Pi 4/5** |
| **MediaPipe Full (complexity=1)** | ~30 MB | 35–55 ms | 20–30 ms | More accurate | Too heavy for 30 FPS on Pi 4 | Avoid |
| **TF Lite Custom Hand Detector (e.g., palm detector + hand landmark)** | 6–15 MB | 18–30 ms | 12–20 ms | Similar to MP but manual pipeline | More code, no benefit over MP | Use only if MP unavailable |
| **YOLOv5-n / YOLOv8-n (object/hand)** | 4–8 MB | 45–80 ms (NCNN) | 25–45 ms | General object detection | Overkill, slower, needs NMS | Not for fingertip |
| **TensorFlow Lite + MoveNet Lightning (pose)** | 8 MB | 25–40 ms | 15–25 ms | Full body | Wrist only, not fingertip | Not precise |

### Decision
**Primary: MediaPipe Hands Lite (complexity=0)** — fastest practical deep model with fingertip precision. Proven to hit ~24–30 FPS on Pi 5 and 20–24 FPS on Pi 4 at 256x192 detect resolution with frame skipping.

**Fallback: Classical HSV Color Tracker** — 10x faster, guarantees 30 FPS on Pi 3B+/Pi 4 1GB. Use bright green/blue glove or tape. HSV is lighting-robust vs RGB.

**Why not pure OpenCV?** Skin-color thresholding without marker is unreliable across skin tones/lighting and jitters heavily.

**Why not TFLite YOLO?** Adds NMS, anchor decoding, higher latency for same task.

## 3. System Architecture

```
                ┌─────────────────────────────────┐
Camera (libcamera/V4L2 MJPG 640x480@30) ──┤ Threaded Capture (queue=1)  ├─ latest frame ─┐
                └─────────────────────────────────┘                      │                 ▼
                                                                  ┌─────────────┐   ┌──────────────┐
                                                                  │  Main Loop  │   │  Downscale   │
                                                                  │  640x480    │◀──│ 320x240 /    │
                                                                  └──────┬──────┘   │ 256x192 LANCZ│
                                                                         │          └──────┬───────┘
                                                           ┌─────────────┼────────────────┘
                                                           ▼             │
                                                   ┌──────────────┐      ▼
                                                   │  Detector    │  RGB vs BGR
                                                   │ MP Lite OR   │  (MP needs RGB)
                                                   │ HSV Contour  │
                                                   └──────┬───────┘
                                                          │ (x,y) detect space
                                                          ▼
                                                   ┌──────────────┐
                                                   │  Scale to    │
                                                   │  cam space   │
                                                   │  + One-Euro  │
                                                   │  Smoother    │
                                                   └──────┬───────┘
                                                          │ (x,y) cam space
                                                          ▼
                                                   ┌──────────────┐
                                                   │ VirtualLight │
                                                   │ Precomputed  │
                                                   │ Radial Glow  │
                                                   │ + Alpha/Add  │
                                                   └──────┬───────┘
                                                          │
                                                          ▼
                                                   ┌──────────────┐
                                                   │  Display     │
                                                   │  FPS/Latency │
                                                   │  Overlay     │
                                                   └──────────────┘
```

### Modules
- `config.py`: Profiles (pi5/pi4/pi4_low/pc_debug), tunables
- `tracker.py`: HybridTracker (MP + Color), Smoother, latency measurement
- `virtual_light.py`: Precomputed texture, ROI blending
- `main.py`: ThreadedCamera, main loop, metrics, key handling

## 4. Performance Budget (Pi 4 @ 640x480 display, 256x192 detect, skip=2)

| Stage | Time | % | Notes |
|---|---|---|---|
| Capture (threaded, off critical path) | 0 ms (async) | 0 | Queue=1 ensures no buffering |
| Resize 640→256 | ~0.6 ms | 5 | INTER_LINEAR + NEON |
| Detect (MediaPipe Lite) | ~16 ms avg | 55 | Only every 2nd frame → effective 8ms/frame |
| Smooth + scale | ~0.1 ms | 1 | NumPy |
| Render glow (ROI blend) | ~0.8 ms | 7 | Precomputed, ROI only |
| Display + overlay | ~2 ms | 12 | |
| **Total per displayed frame** | **~12–18 ms** | — | → **28–40 FPS feasible** |
| End-to-end latency | **~35–55 ms** | — | Capture + process + display |

With `pi4_low` (color backend): total ~5–8 ms → 60+ FPS possible but limited by camera 30fps.

## 5. Optimization Techniques (why each helps)

1. **Threaded capture + queue=1**: Decouples V4L2 blocking read (can stall 33ms) from processing. Queue=1 drops stale frames → reduces latency by ~30ms vs buffered capture.
2. **MJPG fourcc + V4L2 + Buffersize=1**: MJPEG is hardware-compressed on Pi camera, 3x faster than YUYV raw. Buffersize 1 avoids driver queueing 3–4 frames.
3. **Picamera2 (libcamera)**: Zero-copy DMA path on Pi 5/CM3, lower latency than V4L2 shim.
4. **Downscaled detection**: 640x480=307k px vs 256x192=49k px → 6.2x fewer pixels for detector (linear speedup, ~6x less memory bandwidth).
5. **Frame skipping (detect every N)**: Detector is bottleneck (16ms). Skipping 1 frame halves detector load → effective 8ms/frame overhead, still tracks smoothly due to interpolation + smoother.
6. **Precomputed glow texture**: Per-frame GaussianBlur(160x160) would be ~8ms/frame. Precomputing once saves 40% budget. Runtime is just ROI memcpy + blend.
7. **In-place ROI blending**: No full-frame copy (640x480x3 = 0.9MB per frame). Only touching 100x100–160x160 ROI → 10x less memory traffic.
8. **Additive vs alpha blend**: `cv2.add` is NEON-accelerated and saturates correctly for glow without float conversion.
9. **NEON / OpenCV optimized**: `cv2.setUseOptimized(True)` enables ARM NEON SIMD. `cv2.setNumThreads(2)` prevents thread contention (Pi has 4 cores: leave 2 for app/camera).
10. **Smoother adaptive alpha**: One-Euro style: low speed → heavy smoothing (reduces jitter), high speed → low smoothing (reduces lag). Avoids Kalman overhead.
11. **Grayscale/HSV morph kernel reuse**: Pre-allocated kernel, avoid per-frame alloc.
12. **RGB conversion only for MP**: Color backend stays BGR → saves one cvt every N frames.
13. **Flip once**: Mirror effect via single `cv2.flip` (cheap) rather than coordinate remapping.

## 6. Camera Resolution & FPS Recommendations

| Pi Model | Camera Res | Detect Res | Target FPS | Notes |
|---|---|---|---|---|
| Pi 5 (8GB) | 640x480 MJPG | 320x240 | 30 | Use Picamera2 if Camera Module 3, else V4L2 USB |
| Pi 5 (4GB) | 640x480 | 320x240 | 28–30 | Same, complexity 0 |
| Pi 4 (4GB/8GB) | 640x480 | 256x192 | 24–28 | Skip=2, still smooth |
| Pi 4 (2GB) | 480x360 | 256x192 | 24 | Lower display res to save memory bw |
| Pi 4 (1GB) / Pi 3B+ | 480x360 | 192x144 | 20 | Backend=color mandatory |
| Pi Zero 2 W | 480x360 | 192x144 | 15–20 | Color only, skip=3 |

Avoid 1280x720: 2.25x pixels, detector must downscale more, no visual benefit for light effect.

## 7. Latency Reduction Checklist

- [ ] Use MJPG not YUYV (`V4L2_FOURCC`)
- [ ] Set `CAP_PROP_BUFFERSIZE=1`
- [ ] Threaded capture with queue=1 (latest frame)
- [ ] Disable autofocus/exposure hunting if camera supports
- [ ] Small detect resolution (256x192)
- [ ] Frame skipping + smoothing interpolation
- [ ] Precomputed textures
- [ ] `cv2.waitKey(1)` not 30
- [ ] Run without desktop compositor (console/kms) saves ~8ms
- [ ] Overclock Pi 4 to 1.8GHz if cooling allows (`arm_freq=1800` in config.txt)

## 8. Glow Rendering Technique

- Radial gradient: `falloff = (1 - dist/r)^1.8`, blurred with σ=0.35*r
- Precompute BGR texture (uint8) + alpha float32 3ch
- Per-frame: compute ROI clipped to frame bounds, blend via `cv2.add(roi, tex*alpha)`
- Inner core: two `cv2.circle` (white + color) for bulb illusion (0.05ms)
- No per-frame `GaussianBlur`, no full-screen alpha.

Alternatives considered: shader-based glow (needs OpenGL), particle system (heavy).

## 9. Pi 4 vs Pi 5 Specifics

- **Pi 5**: Cortex-A76 2.4GHz, VideoCore VII, PCIe, faster memory (4267 MT/s). MediaPipe Lite 40% faster. Can handle 320x240 detect at every frame, 80px glow.
- **Pi 4**: Cortex-A72 1.5GHz, VideoCore VI. Memory bandwidth is bottleneck → lower detect res and skip frames. Active cooling recommended (throttling at 80°C drops 20% FPS).
- Both: Use 64-bit Bookworm, `arm_boost=1` on Pi4, enable `dtoverlay=imx708` for Camera Module 3.

## 10. Failure Modes & Fallbacks

- No hand → hide light, show hint, keep last position for <5 frames then hide (avoid ghost)
- MP not installed → auto fallback to color
- Camera open fail → try index 0,1,2 then exit with hint
- Low FPS (<15) → auto increase skip from 2→3 and reduce glow radius (adaptive)
