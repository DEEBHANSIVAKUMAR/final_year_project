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
import os
import threading
import queue
import argparse
import sys
# --- PROOF THAT THIS FILE IS RUNNING (STEP 1) ---
print(f"RUNNING FILE: {os.path.abspath(__file__)}")
print("VERSION: DIRECTION_DEBUG_V2")
# Also write to a marker file for verification
try:
    with open(os.path.join(os.path.dirname(__file__), "running_version.txt"), "w") as f:
        f.write(f"RUNNING FILE: {os.path.abspath(__file__)}\nVERSION: DIRECTION_DEBUG_V2\n")
except:
    pass
from collections import deque

import config
from config import get_config, TRACKER_BACKEND, SMOOTHING_ALPHA, SMOOTHING_BETA, LIGHT_COLOR_BGR, LIGHT_INTENSITY, LIGHT_BLEND_MODE, V4L2_FOURCC, CAMERA_FPS_REQUEST, LOW_LIGHT_THRESHOLD, FACE_LIGHT_COLOR_BGR, FACE_LIGHT_INTENSITY, ENABLE_HEAD_POSE, HEAD_SENSITIVITY, DEBUG_MODE, PERFORMANCE_MODE, FACE_CALIB_YAW_TOL, FACE_CALIB_PITCH_TOL, FACE_CALIB_ROLL_TOL, DEBUG_DIRECTION, DIRECTION_CONFIRM_FRAMES, BODY_DIRECTION_INVERT
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
    print("  SIMULATION KEYS: 1=LEFT 2=RIGHT 3=FORWARD 4=BACKWARD 0=STOP (verify UI can change)")
    print("  DEBUG_V2 panel shows live FACE YAW/PITCH/ROLL and BODY TURN values for threshold tuning")
    print(f"  BODY_DIRECTION_INVERT={BODY_DIRECTION_INVERT} DIRECTION_CONFIRM_FRAMES={DIRECTION_CONFIRM_FRAMES}")
    # Manual simulation state (STEP 7)
    simulated_command = None
    simulated_until = 0
    # --- PERFORMANCE: cache config values outside loop (avoid dict lookup) ---
    _cam_w = cfg["camera_width"]
    _cam_h = cfg["camera_height"]
    _perf_face_every = cfg.get("face_mesh_every_n", 2)
    if DEBUG_MODE:
        print(f"[Performance] Pi5 optimized: cam {_cam_w}x{_cam_h} -> detect {cfg['detect_width']}x{cfg['detect_height']} face_mesh_every_n={_perf_face_every} DEBUG_MODE={DEBUG_MODE}")
    try:
        while True:
            loop_start = time.perf_counter()
            t0 = loop_start
            frame = cam.read(timeout=1.0)
            t_capture = (time.perf_counter() - t0)*1000
            if frame is None:
                continue

            t1 = time.perf_counter()
            # Ensure correct size (picamera vs V4L2 may differ) - cache check
            if frame.shape[1] != _cam_w or frame.shape[0] != _cam_h:
                frame = cv2.resize(frame, (_cam_w, _cam_h), interpolation=cv2.INTER_LINEAR)
            # Mirror for natural interaction (single flip, no extra copy)
            frame = cv2.flip(frame, 1)
            t_resize_flip = (time.perf_counter() - t1)*1000

            t2 = time.perf_counter()
            # Track - supports both new dict API and legacy (x,y)
            # PERFORMANCE: BGR->RGB conversion is cached inside FaceDirectionTracker; no duplicate here
            res = tracker.update(frame)
            detect_ms = tracker.get_last_detect_latency_ms()
            t_track = (time.perf_counter() - t2)*1000

            t3 = time.perf_counter()
            # Fast brightness estimate for low-light auto boost (~0.2ms) - skip in performance mode if not low light?
            # Use sampled frame[::4] already optimized; keep but cache low_light check
            mean_v = VirtualLight.estimate_brightness(frame)
            low_light = mean_v < LOW_LIGHT_THRESHOLD
            t_brightness = (time.perf_counter() - t3)*1000

            # Render logic: FACE DIRECTION with AUTO-CALIBRATION (small turn -> LEFT/RIGHT)
            rendered = False
            status_text = ""
            head_info = {"vector": (0,0), "direction": "NO FACE", "yaw": 0, "pitch": 0, "calibrated": False, "command": "STOP"}
            primary_vec = (0,0); primary_dir = "CENTER"
            # Track calibration transition for CALIBRATION COMPLETE display
            if 'was_calibrated' not in locals():
                was_calibrated = False
            calib_complete_display_until = locals().get('calib_complete_display_until', 0)
            if isinstance(res, dict):
                face = res.get("face")
                face_center = res.get("face_center")
                hand_pt = res.get("hand")
                head_info = res.get("head_pose", head_info)
                # Use overall calibrated (face+body) if available, else head
                body_info = res.get("body_pose", {})
                primary_vec = res.get("primary_vec", head_info["vector"])
                primary_dir = res.get("primary_dir", head_info["direction"])
                command = res.get("command", "STOP")
                # STEP 7: Manual simulation override (proves UI can change and STOP not hardcoded)
                if simulated_command is not None:
                    if time.time() < simulated_until:
                        command = simulated_command
                        primary_dir = simulated_command
                        primary_vec = {"LEFT": (-0.8,0), "RIGHT": (0.8,0), "FORWARD": (0,-0.8), "BACKWARD": (0,0.8), "STOP": (0,0)}.get(simulated_command, (0,0))
                        gx, gy = primary_vec
                    else:
                        print(f"[Simulation] Ended {simulated_command}, back to vision")
                        simulated_command = None
                gx, gy = primary_vec
                is_calibrated = res.get("calibrated", head_info.get("calibrated", True))
                # Detect calibration just completed
                if is_calibrated and not was_calibrated:
                    calib_complete_display_until = time.time() + 2.0
                    print("[System] CALIBRATION COMPLETE - Direction detection starts")
                was_calibrated = is_calibrated
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
                        # Debug overlay: show why frame is valid/rejected during calibration
                        yaw_dbg = head_info.get("yaw", 0)
                        pitch_dbg = head_info.get("pitch", 0)
                        roll_dbg = head_info.get("roll", 0)
                        conf_dbg = head_info.get("confidence", 0.0)
                        face_dbg = head_info.get("face_detected", face is not None)
                        reason = head_info.get("rejection_reason", "")
                        dbg_text = f"Face:{face_dbg} Yaw:{yaw_dbg:.1f} Pitch:{pitch_dbg:.1f} Roll:{roll_dbg:.1f} Conf:{conf_dbg:.2f} Valid:{reason==''} {reason}"
                        cv2.putText(frame, dbg_text, (8, cfg["camera_height"]-28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,255,255) if face_dbg else (0,165,255), 1, cv2.LINE_AA)
                        if reason:
                            cv2.putText(frame, f"Rejected: {reason}", (cfg["camera_width"]//2 - 110, cfg["camera_height"]//2 + 122),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,255), 1, cv2.LINE_AA)
                        else:
                            cv2.putText(frame, f"Accepted: {prog}/{total}", (cfg["camera_width"]//2 - 60, cfg["camera_height"]//2 + 122),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,0), 1, cv2.LINE_AA)
                    else:
                        # Show CALIBRATION COMPLETE for 2 sec after auto-calibration
                        if time.time() < locals().get('calib_complete_display_until', 0):
                            cv2.putText(frame, "CALIBRATION COMPLETE", (cfg["camera_width"]//2 - 110, cfg["camera_height"]//2 + 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                            status_text = "CALIBRATION COMPLETE - Direction detection starts"
                        # Face turn -> light moves: LEFT -> light LEFT, RIGHT -> light RIGHT (now with calibrated CENTER)
                        # Stable locked direction: only treat LEFT/RIGHT/FORWARD/BACKWARD as turned; STOP/CENTER/CALIBRATING keep centred.
                        is_turned = primary_dir not in ("CENTER", "STOP", "CALIBRATING", "NO FACE", "NO BODY")
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
                        if primary_dir not in ("CENTER", "STOP", "CALIBRATING", "NO FACE"):
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

            # Metrics + PERFORMANCE breakdown
            t_render_end = time.perf_counter()
            t_render = (t_render_end - t_brightness)*1000  # approx includes render logic above
            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0
            fps_hist.append(fps)
            avg_fps = sum(fps_hist) / len(fps_hist)
            latency_ms = (time.perf_counter() - loop_start) * 1000
            # Breakdown for Pi5 bottleneck analysis
            t_total = latency_ms
            # Estimate direction/calibration inside track, drawing/display below
            t_draw_start = time.perf_counter()
            # STEP 3: On-screen DEBUG_V2 panel (must show all required fields)
            # Always show version proof
            cv2.putText(frame, "RUNNING VERSION: DEBUG_V2", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1, cv2.LINE_AA)
            # Gather debug values (handle missing keys gracefully)
            try:
                body_dbg = body_info if 'body_info' in locals() else {}
                face_yaw = head_info.get("yaw", 0)
                face_pitch = head_info.get("pitch", 0)
                face_roll = head_info.get("roll", 0)
                left_sho = body_dbg.get("left_shoulder", (0,0,0))
                right_sho = body_dbg.get("right_shoulder", (0,0,0))
                # shoulder depth difference and body turn value (normalized)
                shoulder_depth_diff = body_dbg.get("shoulder_depth_diff", 0.0)
                if shoulder_depth_diff==0 and isinstance(left_sho,tuple) and len(left_sho)==3:
                    shoulder_depth_diff = float(right_sho[2] - left_sho[2])
                body_turn = body_dbg.get("body_turn_value", body_dbg.get("body_angle", 0))
                if body_turn==0 and "normalized_turn" in str(body_dbg):
                    body_turn = body_dbg.get("normalized_turn", 0)
                raw_face = head_info.get("raw", head_info.get("candidate_direction", head_info.get("direction","STOP")))
                raw_body = body_dbg.get("raw", body_dbg.get("candidate_direction", body_dbg.get("direction","STOP")))
                # Candidate and counts
                cand_dir = body_dbg.get("candidate_direction", body_dbg.get("raw", primary_dir)) if body_dbg.get("pose_landmarks") else head_info.get("candidate_direction", head_info.get("raw", primary_dir))
                # Prefer body candidate if body is primary
                if 'body_info' in locals() and body_dbg.get("pose_landmarks"):
                    cand_count = body_dbg.get("candidate_count", body_dbg.get("pending_frames", 0))
                    cand_thresh = body_dbg.get("confirm_threshold", DIRECTION_CONFIRM_FRAMES)
                else:
                    cand_count = head_info.get("candidate_count", head_info.get("pending_frames", 0))
                    cand_thresh = head_info.get("confirm_threshold", DIRECTION_CONFIRM_FRAMES)
                confirmed = command
                # STOP reason
                stop_reason = head_info.get("stop_reason", head_info.get("rejection_reason",""))
                if not stop_reason:
                    stop_reason = body_dbg.get("stop_reason", body_dbg.get("rejection_reason",""))
                if command != "STOP":
                    stop_reason = ""
                elif not stop_reason:
                    stop_reason = "threshold not reached"
                # Build panel background
                panel_x, panel_y, panel_w, panel_h = 8, 80, 320, 210
                overlay = frame.copy()
                cv2.rectangle(overlay, (panel_x, panel_y), (panel_x+panel_w, panel_y+panel_h), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                y0 = panel_y + 14
                line_h = 13
                def put(txt, idx, col=(200,255,200)):
                    cv2.putText(frame, txt, (panel_x+6, y0+idx*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
                put(f"FACE DETECTED: {head_info.get('face_detected', face is not None)}", 0)
                put(f"POSE DETECTED: {body_dbg.get('pose_landmarks', False)}", 1, (0,255,0) if body_dbg.get("pose_landmarks") else (0,0,255))
                put(f"FACE YAW: {face_yaw:.1f} PITCH: {face_pitch:.1f} ROLL: {face_roll:.1f}", 2)
                put(f"LEFT SHOULDER: {left_sho[0]:.2f},{left_sho[1]:.2f},{left_sho[2]:.2f}" if isinstance(left_sho,tuple) else f"LEFT SHOULDER: {left_sho}", 3)
                put(f"RIGHT SHOULDER: {right_sho[0]:.2f},{right_sho[1]:.2f},{right_sho[2]:.2f}" if isinstance(right_sho,tuple) else f"RIGHT SHOULDER: {right_sho}", 4)
                put(f"SHOULDER DEPTH DIFF: {shoulder_depth_diff:.3f}", 5)
                put(f"BODY TURN VALUE: {body_turn:.3f} (norm)", 6)
                put(f"RAW FACE DIR: {raw_face}", 7)
                put(f"RAW BODY DIR: {raw_body}", 8)
                put(f"CANDIDATE DIR: {cand_dir}  COUNT: {cand_count}/{cand_thresh}", 9, (0,255,255) if cand_dir!="STOP" else (180,180,180))
                put(f"CONFIRMED DIR: {confirmed}", 10, (0,255,0) if confirmed!="STOP" else (0,0,255))
                put(f"FINAL COMMAND: {command}", 11, (0,255,0) if command!="STOP" else (0,165,255))
                if simulated_command is not None:
                    put(f"SIMULATED: {simulated_command}", 12, (255,255,0))
                else:
                    put(f"STOP REASON: {stop_reason}", 12, (0,0,255) if command=="STOP" else (200,200,200))
                put(f"BODY_INVERT={config.BODY_DIRECTION_INVERT} CONFIRM={config.DIRECTION_CONFIRM_FRAMES}", 13, (180,180,180))
                put(f"CALIB YAW:{head_info.get('yaw',0):.1f} BODY_YAW:{body_dbg.get('yaw',0):.1f}", 14)
                put(f"NEUTRAL YAW:{head_info.get('rel_yaw',0):.1f} if cal", 15)
            except Exception as e:
                if DEBUG_MODE:
                    print(f"[DebugPanel] error {e}")
            t_draw_start = time.perf_counter()
            # Overlay status for debug - reduce in PERFORMANCE_MODE after calibrated
            do_debug_overlay = DEBUG_MODE or (head_info.get("calibrated")==False)
            if do_debug_overlay and 'status_text' in locals() and status_text:
                cv2.putText(frame, status_text, (8, _cam_h-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,255,200), 1, cv2.LINE_AA)
            # Display-only: do not wire this directly to motors.
            if do_debug_overlay and isinstance(res, dict):
                # Use already simulated command, not raw res
                disp_cmd = command
                command_color = (0, 255, 0) if disp_cmd != "STOP" else (0, 0, 255)
                cv2.putText(frame, f"WHEELCHAIR COMMAND: {disp_cmd}", (8, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.62, command_color, 2, cv2.LINE_AA)
            # Face-direction compass overlay - skip in PERFORMANCE_MODE when calibrated to save ~2ms drawing
            show_compass = ENABLE_HEAD_POSE and isinstance(res, dict) and (not PERFORMANCE_MODE or not head_info.get("calibrated", True) or DEBUG_MODE)
            if show_compass:
                cx = _cam_w//2
                cy = _cam_h-42
                cv2.circle(frame, (cx, cy), 26, (80,80,80), 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 4, (255,255,255), -1, cv2.LINE_AA)
                if 'head_info' in locals():
                    hx, hy = head_info["vector"]
                    dx = int(hx * 15)
                    dy = int(hy * 15)
                    col = (255,255,255) if head_info["direction"] not in ("CENTER", "STOP") else (180,180,180)
                    cv2.circle(frame, (cx+dx, cy+dy), 7, col, -1, cv2.LINE_AA)
                    cv2.circle(frame, (cx+dx, cy+dy), 7, (0,180,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "L", (cx-40, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "R", (cx+32, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "U", (cx-5, cy-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "D", (cx-5, cy+40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
                cv2.putText(frame, "FACE DIR", (cx-24, cy+54), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200,200,200), 1, cv2.LINE_AA)
            elif isinstance(res, dict) and PERFORMANCE_MODE:
                # Minimal command text only in performance mode (1 putText instead of 5+)
                command = res.get("command", "STOP")
                if command != "STOP":
                    cv2.putText(frame, f"CMD:{command}", (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2, cv2.LINE_AA)
            t_draw = (time.perf_counter() - t_draw_start)*1000
            # Always show FPS bar (cheap)
            draw_metrics(frame, avg_fps, latency_ms, detect_ms, cfg["profile_name"], tracker.backend_name)
            t_display_start = time.perf_counter()
            cv2.imshow("Virtual Light - Face Fill (q=quit, c=color, m=backend, f=face/hand)", frame)
            t_display = (time.perf_counter() - t_display_start)*1000
            # Periodic performance log (~every 30 frames) and detailed calibration already throttled
            if frame_count % 30 == 0:
                if DEBUG_MODE or PERFORMANCE_MODE:
                    print(f"[PERFORMANCE] capture={t_capture:.1f}ms resize_flip={t_resize_flip:.1f}ms track={t_track:.1f}ms brightness={t_brightness:.1f}ms render~{t_render:.1f}ms drawing={t_draw:.1f}ms display={t_display:.1f}ms total={t_total:.1f}ms fps={avg_fps:.1f}")
                # Also log calibration fallback thresholds for visibility
                if not head_info.get("calibrated", True):
                    print(f"[Calibration] progress {head_info.get('calib_progress',0)}/{head_info.get('calib_total',25)} yaw_tol={FACE_CALIB_YAW_TOL} pitch_tol={FACE_CALIB_PITCH_TOL} roll_tol={FACE_CALIB_ROLL_TOL} fallback={cfg.get('face_mesh_every_n',2)}")

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
            elif key == ord('1'):
                simulated_command = "LEFT"
                simulated_until = time.time() + 5
                print("[Simulation] LEFT for 5s - UI should show LEFT (TEST A)")
            elif key == ord('2'):
                simulated_command = "RIGHT"
                simulated_until = time.time() + 5
                print("[Simulation] RIGHT for 5s - UI should show RIGHT (TEST B)")
            elif key == ord('3'):
                simulated_command = "FORWARD"
                simulated_until = time.time() + 5
                print("[Simulation] FORWARD for 5s")
            elif key == ord('4'):
                simulated_command = "BACKWARD"
                simulated_until = time.time() + 5
                print("[Simulation] BACKWARD for 5s")
            elif key == ord('0'):
                simulated_command = "STOP"
                simulated_until = time.time() + 5
                print("[Simulation] STOP for 5s")
            elif key == ord('b'):
                # Toggle BODY_DIRECTION_INVERT to test sign
                import config as cfg_mod
                cfg_mod.BODY_DIRECTION_INVERT = not cfg_mod.BODY_DIRECTION_INVERT
                # Also update tracker instances if needed (they read config directly each frame, so next frame will use new value)
                print(f"[Config] BODY_DIRECTION_INVERT toggled to {cfg_mod.BODY_DIRECTION_INVERT} - turn LEFT to test mapping")

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
