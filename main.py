"""
main.py - Real-time Virtual Light System for Raspberry Pi
Priorities: Real-Time Performance > Low Latency > Stable Tracking > Visual Quality

Architecture:
  Camera Thread (producer) -> Main Thread (consumer: detect + render + display)
  Uses queue size 1 (always latest frame) to minimize latency.

Optimizations implemented:
  - Threaded capture (decouples capture from processing)
  - Picamera2 support (zero-copy on Pi) with V4L2 fallback
  - MJPG fourcc, buffered queue=1
  - Downscaled detection (256x192 / 320x240) -> 4-6x fewer pixels
  - Frame skipping (detect every N frames, interpolate)
  - Precomputed glow texture (no per-frame blur)
  - In-place ROI blending (no full-frame copy)
  - cv2.UMat / NEON auto-accel where available
  - Adaptive FPS display & latency measurement
"""
import cv2
import time
import threading
import queue
import argparse
import sys
from collections import deque

from config import get_config, TRACKER_BACKEND, SMOOTHING_ALPHA, SMOOTHING_BETA, LIGHT_COLOR_BGR, LIGHT_INTENSITY, LIGHT_BLEND_MODE, V4L2_FOURCC, CAMERA_FPS_REQUEST, LOW_LIGHT_THRESHOLD, FACE_LIGHT_COLOR_BGR, FACE_LIGHT_INTENSITY, ENABLE_HEAD_POSE, HEAD_SENSITIVITY
from tracker import HybridTracker
from virtual_light import VirtualLight

# Try Picamera2
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class ThreadedCamera:
    """Low-latency threaded camera with latest-frame queue (size 1)"""
    def __init__(self, cfg, use_picamera2=True):
        self.cfg = cfg
        self.w = cfg["camera_width"]
        self.h = cfg["camera_height"]
        self.use_picamera2 = use_picamera2 and PICAMERA2_AVAILABLE and cfg.get("use_picamera2", False)
        self.queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.cap = None
        self.picam2 = None

    def start(self):
        self.running = True
        if self.use_picamera2:
            print("[Camera] Using Picamera2 (hardware accelerated)")
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (self.w, self.h), "format": "RGB888"},
                controls={"FrameRate": CAMERA_FPS_REQUEST}
            )
            self.picam2.configure(config)
            self.picam2.start()
            # No extra thread needed for picamera2? Use polling thread for uniform API
            self.thread = threading.Thread(target=self._picam_loop, daemon=True)
            self.thread.start()
        else:
            print(f"[Camera] Using V4L2 / OpenCV (MJPG {self.w}x{self.h} @ {CAMERA_FPS_REQUEST}fps)")
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)  # fallback
            self.cap.set(cv2.CAP_PROP_FOURCC, V4L2_FOURCC)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS_REQUEST)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency
            # Try to disable auto-exposure lag on Pi
            # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera")
            # Warmup
            for _ in range(5):
                self.cap.read()
            self.thread = threading.Thread(target=self._v4l2_loop, daemon=True)
            self.thread.start()
        return self

    def _v4l2_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Drop old frame if queue full (keep latest)
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def _picam_loop(self):
        while self.running:
            frame = self.picam2.capture_array()
            # Picamera2 gives RGB, convert to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)
            # Small sleep to avoid 100% CPU
            time.sleep(0.001)

    def read(self, timeout=1.0):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        if self.picam2:
            try:
                self.picam2.stop()
            except:
                pass


def draw_metrics(frame, fps, latency_ms, detect_ms, profile, backend):
    """Minimal overlay - cheap text rendering"""
    h, w = frame.shape[:2]
    # Semi-transparent bar at top
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    # FPS color: green if >=20, yellow if 15-20, red if <15
    fps_color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 15 else (0, 0, 255)
    text = f"FPS:{fps:.1f} | Lat:{latency_ms:.1f}ms | Det:{detect_ms:.1f}ms | {profile} | {backend} | Q:Quit C:Color M:MP"
    cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # FPS dot
    cv2.circle(frame, (w - 16, 16), 8, fps_color, -1, lineType=cv2.LINE_AA)
    return frame


def parse_args():
    p = argparse.ArgumentParser(description="Virtual Light - Pi Optimized (Face Fill Light for low-light)")
    p.add_argument("--profile", choices=["pi5", "pi4", "pi4_low", "pc_debug"], default=None, help="Hardware profile")
    p.add_argument("--backend", choices=["face", "mediapipe", "color", "auto"], default=None, help="Tracker backend: face=fill-light (default, perfect for low light), mediapipe=hand orb, color=HSV, auto=face->hand")
    p.add_argument("--mode", choices=["face", "hand", "auto"], default=None, help="Alias for backend - face mode glows on face detection")
    p.add_argument("--face-detector", choices=["haar", "mediapipe", "auto"], default="auto", help="Face detector: haar (~2ms, default) or mediapipe")
    p.add_argument("--camera", type=int, default=0, help="Camera index (V4L2)")
    p.add_argument("--no-picamera2", action="store_true", help="Force V4L2 even if picamera2 available")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.profile)
    # --mode alias for --backend
    backend = args.backend or args.mode or TRACKER_BACKEND
    # normalize hand alias
    if backend == "hand":
        backend = "mediapipe"

    print("="*60)
    print(f" Virtual Light System - Face Fill Light Edition")
    print(f" Profile: {cfg['profile_name']} | Backend: {backend} | FaceDetector: {args.face_detector}")
    print(f" Camera: {cfg['camera_width']}x{cfg['camera_height']} -> Detect: {cfg['detect_width']}x{cfg['detect_height']}")
    print(f" Target FPS: {cfg['fps_target']} | Skip: every {cfg['detect_every_n_frames']} frame(s)")
    print(f" Low-light threshold: {LOW_LIGHT_THRESHOLD} | Fill-light will auto-boost in dark")
    print("="*60)

    # Enable OpenCV optimizations
    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)  # Pi has 4 cores: 2 for OpenCV, 1 for camera, 1 for main

    # Camera
    use_picam = not args.no_picamera2
    cam = ThreadedCamera(cfg, use_picamera2=use_picam).start()
    # Wait for first frame
    frame = None
    for _ in range(30):
        frame = cam.read(timeout=1.0)
        if frame is not None:
            break
        print("[Camera] Waiting for frame...")
        time.sleep(0.1)
    if frame is None:
        print("[Error] No camera frame received. Check camera connection.")
        cam.stop()
        sys.exit(1)

    # Tracker + Light
    tracker = HybridTracker(cfg=cfg, backend=backend, smoother_alpha=SMOOTHING_ALPHA, smoother_beta=SMOOTHING_BETA, face_detector=args.face_detector)
    light = VirtualLight(radius=cfg["light_radius"], color_bgr=LIGHT_COLOR_BGR, intensity=LIGHT_INTENSITY, mode=LIGHT_BLEND_MODE)
    # Dedicated face fill-light (warmer, softer) - shares texture logic but different color/intensity
    face_light = VirtualLight(radius=int(cfg["light_radius"]*1.4), color_bgr=FACE_LIGHT_COLOR_BGR, intensity=FACE_LIGHT_INTENSITY, mode=LIGHT_BLEND_MODE)

    # Metrics
    fps_hist = deque(maxlen=30)
    prev_time = time.perf_counter()
    # FPS calc
    frame_count = 0

    # For key handling: allow switching light color - WHITE first (pure white, not warm)
    colors = [
        (255, 255, 255),  # pure white (default)
        (255, 255, 240),  # soft white
        (200, 220, 255),  # cool white
        (255, 255, 255),  # white boost (duplicate for cycling)
        (0, 220, 255),    # warm (old)
    ]
    color_idx = 0

    print("[System] Running - calibrated Face Mesh direction detection")
    print("  q=quit | r=recalibrate CENTER | c=color | m=toggle backend | f=face/hand mode | s=snapshot")
    print("  Keep face straight during calibration. Commands: LEFT / RIGHT / FORWARD / BACKWARD / STOP")
    print("  Face glow auto-boosts in low light (<{} V)".format(LOW_LIGHT_THRESHOLD))
    try:
        while True:
            loop_start = time.perf_counter()
            frame = cam.read(timeout=1.0)
            if frame is None:
                continue

            # Ensure correct size (picamera vs V4L2 may differ)
            if frame.shape[1] != cfg["camera_width"] or frame.shape[0] != cfg["camera_height"]:
                frame = cv2.resize(frame, (cfg["camera_width"], cfg["camera_height"]), interpolation=cv2.INTER_LINEAR)

            # Mirror for natural interaction
            frame = cv2.flip(frame, 1)

            # Track - supports both new dict API and legacy (x,y)
            res = tracker.update(frame)
            detect_ms = tracker.get_last_detect_latency_ms()

            # Fast brightness estimate for low-light auto boost (~0.2ms)
            mean_v = VirtualLight.estimate_brightness(frame)
            low_light = mean_v < LOW_LIGHT_THRESHOLD

            # Render logic: FACE DIRECTION with AUTO-CALIBRATION (small turn -> LEFT/RIGHT)
            rendered = False
            status_text = ""
            head_info = {"vector": (0,0), "direction": "NO FACE", "yaw": 0, "pitch": 0, "calibrated": False, "command": "STOP"}
            primary_vec = (0,0); primary_dir = "CENTER"
            if isinstance(res, dict):
                face = res.get("face")
                face_center = res.get("face_center")
                hand_pt = res.get("hand")
                head_info = res.get("head_pose", head_info)
                # FaceDirectionTracker is primary - auto-calibrated CENTER
                primary_vec = res.get("primary_vec", head_info["vector"])
                primary_dir = res.get("primary_dir", head_info["direction"])
                command = res.get("command", "STOP")
                gx, gy = primary_vec
                is_calibrated = head_info.get("calibrated", True)
                if face is not None or face_center is not None:
                    use_center = face_center if face_center is not None else (face[0]+face[2]//2, face[1]+face[3]//2) if face else None
                    # If calibrating, show progress and keep CENTER
                    if not is_calibrated:
                        face_light.render_face(frame, face, use_center, brightness=mean_v, low_light=low_light)
                        prog = head_info.get("calib_progress", 0)
                        total = head_info.get("calib_total", 25)
                        pct = int(prog/total*100) if total else 0
                        status_text = f"CALIBRATING... keep face straight {prog}/{total} ({pct}%)"
                        cv2.putText(frame, "CALIBRATING... KEEP FACE STRAIGHT", (cfg["camera_width"]//2 - 150, cfg["camera_height"]//2 + 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,255,255), 2, cv2.LINE_AA)
                        # Progress bar
                        bar_w = int(200 * pct/100)
                        cv2.rectangle(frame, (cfg["camera_width"]//2 - 100, cfg["camera_height"]//2 + 95), (cfg["camera_width"]//2 + 100, cfg["camera_height"]//2 + 105), (80,80,80), -1, cv2.LINE_AA)
                        cv2.rectangle(frame, (cfg["camera_width"]//2 - 100, cfg["camera_height"]//2 + 95), (cfg["camera_width"]//2 - 100 + bar_w, cfg["camera_height"]//2 + 105), (0,255,0), -1, cv2.LINE_AA)
                    else:
                        # Face turn -> light moves: LEFT -> light LEFT, RIGHT -> light RIGHT (now with calibrated CENTER)
                        is_turned = primary_dir != "CENTER"
                        if use_center is not None and ENABLE_HEAD_POSE and is_turned:
                            ox = int(gx * HEAD_SENSITIVITY)
                            oy = int(gy * HEAD_SENSITIVITY)
                            gaze_pos = (use_center[0] + ox, use_center[1] + oy)
                            gaze_pos = (max(0, min(cfg["camera_width"]-1, gaze_pos[0])), max(0, min(cfg["camera_height"]-1, gaze_pos[1])))
                            face_light.render_face(frame, face, use_center, brightness=mean_v, low_light=low_light)
                            light.render(frame, gaze_pos)
                            cv2.arrowedLine(frame, use_center, gaze_pos, (255,255,255), 2, cv2.LINE_AA, tipLength=0.22)
                            cv2.circle(frame, gaze_pos, 10, (255,255,255), 2, cv2.LINE_AA)
                        else:
                            face_light.render_face(frame, face, use_center, brightness=mean_v, low_light=low_light)
                        rendered = True
                        calib_str = "" if is_calibrated else " (calib...)"
                        status_text = f"FACE | Pose:{primary_dir} | Command:{command}{calib_str} | V:{mean_v:.0f}"
                        if primary_dir != "CENTER":
                            cv2.putText(frame, primary_dir, (cfg["camera_width"]//2 - 64, cfg["camera_height"]//2 + 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3, cv2.LINE_AA)
                            cv2.putText(frame, primary_dir, (cfg["camera_width"]//2 - 64, cfg["camera_height"]//2 + 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,180,255), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(frame, "CENTER - face straight (r=recalibrate)", (cfg["camera_width"]//2 - 132, cfg["camera_height"]//2 + 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,255,200), 1, cv2.LINE_AA)
                    rendered = True
                    if not is_calibrated:
                        pass  # already set status
                    # Show calib point tiny dot for debug
                    calib = head_info.get("calib", (None,None))
                    if calib and calib[0] is not None and is_calibrated:
                        # small cross at calibrated CENTER
                        cx0, cy0 = int(calib[0]), int(calib[1])
                        cv2.drawMarker(frame, (cx0, cy0), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1, line_type=cv2.LINE_AA)
                elif hand_pt is not None:
                    light.render(frame, hand_pt)
                    rendered = True
                    status_text = f"HAND orb | DIR:{primary_dir} | V:{mean_v:.0f}"
                else:
                    if low_light:
                        face_light._ambient_fill(frame, alpha=0.13)
                        status_text = f"LOW-LIGHT fill (no face) | V:{mean_v:.0f}"
                    else:
                        status_text = f"No face - show your face | V:{mean_v:.0f}"
                    cv2.putText(frame, status_text, (cfg["camera_width"]//2 - 130, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255) if low_light else (0,0,255), 1, cv2.LINE_AA)
            else:
                pt = res
                if pt is not None:
                    light.render(frame, pt)
                    rendered = True
                    status_text = f"HAND | V:{mean_v:.0f}"
                else:
                    if low_light:
                        face_light._ambient_fill(frame, alpha=0.13)
                        status_text = f"LOW-LIGHT fill | V:{mean_v:.0f}"
                    cv2.putText(frame, "No hand/face detected" if not low_light else "Low light - move closer",
                                (cfg["camera_width"]//2 - 110, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            # Metrics
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0
            fps_hist.append(fps)
            avg_fps = sum(fps_hist) / len(fps_hist)
            latency_ms = (time.perf_counter() - loop_start) * 1000

            # Overlay status for debug
            if 'status_text' in locals() and status_text:
                cv2.putText(frame, status_text, (8, cfg["camera_height"]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,255,200), 1, cv2.LINE_AA)
            # Display-only: do not wire this directly to motors. A tested motor
            # controller, emergency-stop switch, and manual override are required.
            if isinstance(res, dict):
                command = res.get("command", "STOP")
                command_color = (0, 255, 0) if command != "STOP" else (0, 0, 255)
                cv2.putText(frame, f"WHEELCHAIR COMMAND: {command}", (8, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, command_color, 2, cv2.LINE_AA)
            # Face-direction compass overlay (turn LEFT -> dot moves LEFT, shows LEFT)
            if ENABLE_HEAD_POSE and isinstance(res, dict):
                cx = cfg["camera_width"]//2
                cy = cfg["camera_height"]-42
                cv2.circle(frame, (cx, cy), 26, (80,80,80), 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 4, (255,255,255), -1, cv2.LINE_AA)
                if 'head_info' in locals():
                    hx, hy = head_info["vector"]
                    dx = int(hx * 15)
                    dy = int(hy * 15)
                    col = (255,255,255) if head_info["direction"] != "CENTER" else (180,180,180)
                    cv2.circle(frame, (cx+dx, cy+dy), 7, col, -1, cv2.LINE_AA)
                    cv2.circle(frame, (cx+dx, cy+dy), 7, (0,180,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "L", (cx-40, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "R", (cx+32, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "U", (cx-5, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "D", (cx-5, cy+40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "FACE DIR", (cx-24, cy+54), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200,200,200), 1, cv2.LINE_AA)

            draw_metrics(frame, avg_fps, latency_ms, detect_ms, cfg["profile_name"], tracker.backend_name)

            cv2.imshow("Virtual Light - Face Fill (q=quit, c=color, m=backend, f=face/hand)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('r'):
                # Recalibrate face CENTER - fixes drift, auto-calibrate again
                tracker.reset_calibration()
                print("[System] Recalibrating - keep face straight...")
            elif key == ord('c'):
                color_idx = (color_idx + 1) % len(colors)
                light.update_color(colors[color_idx])
                face_light.update_color(colors[color_idx])
                print(f"[Light] Color -> {colors[color_idx]}")
            elif key == ord('f'):
                # Toggle face vs hand mode
                new_backend = "mediapipe" if tracker.backend_name == "face" else "face"
                print(f"[Tracker] Switching to {new_backend}...")
                tracker.close()
                try:
                    tracker = HybridTracker(cfg=cfg, backend=new_backend, smoother_alpha=SMOOTHING_ALPHA, smoother_beta=SMOOTHING_BETA, face_detector=args.face_detector)
                except Exception as e:
                    print(f" Switch failed: {e}")
            elif key == ord('m'):
                # Cycle backends
                order = ["face", "mediapipe", "color"]
                try:
                    idx = order.index(tracker.backend_name)
                    new_backend = order[(idx+1) % len(order)]
                except:
                    new_backend = "face"
                print(f"[Tracker] Switching to {new_backend}...")
                tracker.close()
                try:
                    tracker = HybridTracker(cfg=cfg, backend=new_backend, smoother_alpha=SMOOTHING_ALPHA, smoother_beta=SMOOTHING_BETA, face_detector=args.face_detector)
                except Exception as e:
                    print(f" Switch failed: {e}")
            elif key == ord('s'):
                # Save snapshot
                cv2.imwrite(f"snapshot_{int(time.time())}.jpg", frame)
                print("[Camera] Snapshot saved")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[System] Interrupted")

    finally:
        print(f"[System] Avg FPS: {sum(fps_hist)/len(fps_hist) if fps_hist else 0:.1f} over {frame_count} frames")
        tracker.close()
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
