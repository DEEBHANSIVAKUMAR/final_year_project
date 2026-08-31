"""
tracker.py - Lightweight hand/fingertip + FACE tracker
Backends:
  1. Face Haar Cascade (default for virtual fill-light) - ~2ms, no dependency
  2. MediaPipe Hands (Lite) - ~12-25ms on Pi4/5 at 256x192
  3. MediaPipe Face Detection - ~5ms (optional)
  4. Classical HSV Color tracker - ~2-4ms, fallback for low-end Pi

Face returns (x,y, w,h) bbox + center; hand returns (x,y).
"""
import cv2
import numpy as np
import time
import os

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

from config import CFG, MEDIAPIPE_DETECTION_CONF, MEDIAPIPE_TRACKING_CONF, HSV_LOWER, HSV_UPPER, MIN_CONTOUR_AREA, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE_RATIO, ENABLE_HEAD_POSE, HEAD_YAW_THRESHOLD, HEAD_PITCH_THRESHOLD, FACE_AUTO_CALIBRATE, FACE_CALIBRATE_FRAMES, FACE_CALIBRATE_SMOOTHING, HEAD_DIRECTION_HISTORY, COMMAND_CONFIRM_FRAMES, HEAD_DIRECTION_INVERT_X, HEAD_DIRECTION_INVERT_Y


class Smoother:
    """Simple One-Euro / EMA smoother to reduce jitter without lag"""
    def __init__(self, alpha=0.6, beta=0.15):
        self.alpha = alpha
        self.beta = beta
        self.prev = None
        self.velocity = np.zeros(2, dtype=np.float32)

    def update(self, pt):
        """pt: (x,y) or None"""
        if pt is None:
            return None
        pt = np.array(pt, dtype=np.float32)
        if self.prev is None:
            self.prev = pt
            return tuple(pt.astype(int))
        # adaptive alpha based on velocity
        vel = pt - self.prev
        self.velocity = 0.3 * vel + 0.7 * self.velocity
        speed = np.linalg.norm(self.velocity)
        # when moving slowly, smooth more; when fast, be responsive
        adaptive_alpha = np.clip(self.alpha + self.beta * (speed / 50.0), 0.2, 0.95)
        smoothed = adaptive_alpha * pt + (1 - adaptive_alpha) * self.prev
        self.prev = smoothed
        return tuple(smoothed.astype(int))

    def reset(self):
        self.prev = None
        self.velocity = np.zeros(2, dtype=np.float32)


class FaceTrackerHaar:
    """Ultra-light Haar cascade - ships with OpenCV, ~1-3ms at 320x240, no ML model needed"""
    def __init__(self):
        # Use bundled haarcascade - works on Pi and Windows
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            # fallback for opencv-python minimal
            cascade_path = os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade at {cascade_path}")
        # For profile faces (optional second pass)
        self.min_size_ratio = FACE_MIN_SIZE_RATIO

    def detect(self, gray_small):
        """
        gray_small: grayscale image at detect resolution
        returns: (x, y, w, h) bbox in detect-space or None (largest face)
        """
        h, w = gray_small.shape[:2]
        min_h = int(h * self.min_size_ratio)
        min_w = int(w * self.min_size_ratio * 0.8)
        # Equalize for low-light robustness - very cheap, big accuracy win in dark
        eq = cv2.equalizeHist(gray_small)
        faces = self.face_cascade.detectMultiScale(
            eq,
            scaleFactor=FACE_SCALE_FACTOR,
            minNeighbors=FACE_MIN_NEIGHBORS,
            minSize=(min_w, min_h),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) == 0:
            return None
        # Pick largest face (closest person)
        largest = max(faces, key=lambda r: r[2]*r[3])
        return tuple(largest)  # x,y,w,h

class MediaPipeFaceTracker:
    """MediaPipe Face Detection - more accurate than Haar, ~5-8ms"""
    def __init__(self):
        if not MP_AVAILABLE:
            raise RuntimeError("mediapipe not installed")
        self.mp_fd = mp.solutions.face_detection
        self.detector = self.mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    def detect(self, rgb_small):
        results = self.detector.process(rgb_small)
        if not results.detections:
            return None
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w = rgb_small.shape[:2]
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        return (x, y, bw, bh)
    def close(self):
        self.detector.close()


class FaceDirectionTracker:
    """Landmark-based, calibrated head-pose tracker for safe direction detection.

    Face Mesh supplies stable facial landmarks.  ``solvePnP`` estimates the
    rotation of the head, so commands do not depend on the user's body or the
    face box moving inside the camera frame.  No face, calibration, or an
    unstable pose always returns STOP.
    """
    def __init__(self):
        if not MP_AVAILABLE:
            raise RuntimeError("MediaPipe is required for landmark head-pose detection")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.history = []
        self.hist_len = HEAD_DIRECTION_HISTORY
        self.yaw_thresh = float(HEAD_YAW_THRESHOLD)
        self.pitch_thresh = float(HEAD_PITCH_THRESHOLD)
        self.last_vec = (0.0, 0.0)
        self.last_dir = "STOP"
        self.stable_command = "STOP"
        self.pending_command = "STOP"
        self.pending_frames = 0
        # Auto-calibration state
        self.calib_yaw = None
        self.calib_pitch = None
        self.calib_buffer = []
        self.calibrated = False
        self.calib_frames = FACE_CALIBRATE_FRAMES
        self.auto_calibrate = FACE_AUTO_CALIBRATE

    def reset_calibration(self):
        """Manual recalibrate - call when user presses 'r'"""
        self.calib_yaw = None
        self.calib_pitch = None
        self.calib_buffer = []
        self.calibrated = False
        self.history.clear()
        self.last_vec = (0.0, 0.0)
        self.last_dir = "STOP"
        self.stable_command = "STOP"
        self.pending_command = "STOP"
        self.pending_frames = 0
        print("[FaceDirection] Calibration RESET - keep face straight for 1 sec...")

    @staticmethod
    def _rotation_angles(rvec):
        rotation, _ = cv2.Rodrigues(rvec)
        sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
        pitch = np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))
        yaw = np.degrees(np.arctan2(-rotation[2, 0], sy))
        return float(yaw), float(pitch)

    def _measure_pose(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        result = self.face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark
        # Nose, chin, left/right eye corners, left/right mouth corners.
        indices = (1, 152, 33, 263, 61, 291)
        image_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices], dtype=np.float64)
        model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -63.6, -12.5), (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1),
        ], dtype=np.float64)
        camera_matrix = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(model_points, image_points, camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        return self._rotation_angles(rvec)

    def _confirm(self, candidate):
        if candidate == "STOP":
            self.pending_command = "STOP"
            self.pending_frames = 0
            self.stable_command = "STOP"
            return "STOP"
        if candidate == self.pending_command:
            self.pending_frames += 1
        else:
            self.pending_command = candidate
            self.pending_frames = 1
        if self.pending_frames >= COMMAND_CONFIRM_FRAMES:
            self.stable_command = candidate
        return self.stable_command

    def detect(self, frame_bgr, face_bbox=None):
        """Return a calibrated, confirmed wheelchair command from face rotation."""
        if not ENABLE_HEAD_POSE:
            return {"vector": (0, 0), "direction": "STOP", "yaw": 0, "pitch": 0, "calibrated": False, "command": "STOP"}
        pose = self._measure_pose(frame_bgr)
        if pose is None:
            self._confirm("STOP")
            return {"vector": (0, 0), "direction": "NO FACE", "yaw": 0, "pitch": 0, "calibrated": self.calibrated, "command": "STOP"}
        yaw, pitch = pose

        # Auto-calibration logic
        if self.auto_calibrate and not self.calibrated:
            self.calib_buffer.append((yaw, pitch))
            if len(self.calib_buffer) >= self.calib_frames:
                self.calib_yaw = float(np.median([p[0] for p in self.calib_buffer]))
                self.calib_pitch = float(np.median([p[1] for p in self.calib_buffer]))
                self.calibrated = True
                print(f"[FaceDirection] Auto-calibrated CENTER at yaw={self.calib_yaw:.1f}, pitch={self.calib_pitch:.1f}")
                self.history.clear()
            else:
                calib_progress = len(self.calib_buffer)
                self.last_vec = (0.0, 0.0)
                self.last_dir = "CALIBRATING"
                return {"vector": (0, 0), "direction": "CALIBRATING", "yaw": yaw, "pitch": pitch, "calibrated": False, "calib_progress": calib_progress, "calib_total": self.calib_frames, "command": "STOP"}

        relative_yaw = yaw - self.calib_yaw
        relative_pitch = pitch - self.calib_pitch
        if HEAD_DIRECTION_INVERT_X:
            relative_yaw = -relative_yaw
        if HEAD_DIRECTION_INVERT_Y:
            relative_pitch = -relative_pitch
        self.history.append((relative_yaw, relative_pitch))
        if len(self.history) > self.hist_len:
            self.history.pop(0)
        avg_yaw = float(np.mean([p[0] for p in self.history]))
        avg_pitch = float(np.mean([p[1] for p in self.history]))
        self.last_vec = (float(np.clip(avg_yaw / 30, -1, 1)), float(np.clip(avg_pitch / 30, -1, 1)))
        if avg_yaw <= -self.yaw_thresh:
            candidate = "LEFT"
        elif avg_yaw >= self.yaw_thresh:
            candidate = "RIGHT"
        elif avg_pitch <= -self.pitch_thresh:
            candidate = "FORWARD"
        elif avg_pitch >= self.pitch_thresh:
            candidate = "BACKWARD"
        else:
            candidate = "STOP"
            # Adapt only while centered; never adapt during a turn.
            if self.auto_calibrate:
                a = FACE_CALIBRATE_SMOOTHING
                self.calib_yaw = (1 - a) * self.calib_yaw + a * yaw
                self.calib_pitch = (1 - a) * self.calib_pitch + a * pitch
        command = self._confirm(candidate)
        self.last_dir = candidate
        return {"vector": self.last_vec, "direction": candidate, "yaw": avg_yaw, "pitch": avg_pitch, "calibrated": True, "command": command}

    def close(self):
        self.face_mesh.close()


# Backward compat alias - old code expects HeadPoseTracker
HeadPoseTracker = FaceDirectionTracker


class MediaPipeTracker:
    def __init__(self, complexity=0):
        if not MP_AVAILABLE:
            raise RuntimeError("mediapipe not installed")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=complexity,  # 0 = lite (palmdetection lite), 1 = full
            min_detection_confidence=MEDIAPIPE_DETECTION_CONF,
            min_tracking_confidence=MEDIAPIPE_TRACKING_CONF,
        )

    def detect(self, rgb_small):
        """
        rgb_small: RGB image already resized to detect resolution
        returns: (x, y) in detect-space, or None
        """
        # MediaPipe expects RGB, no copy if already RGB
        results = self.hands.process(rgb_small)
        if not results.multi_hand_landmarks:
            return None
        # Use index fingertip (landmark 8) - most intuitive for virtual light
        lm = results.multi_hand_landmarks[0].landmark[8]
        h, w = rgb_small.shape[:2]
        x = int(lm.x * w)
        y = int(lm.y * h)
        return (x, y)

    def close(self):
        self.hands.close()


class ColorTracker:
    """Classical HSV threshold + contour centroid - ultra lightweight"""
    def __init__(self, lower=HSV_LOWER, upper=HSV_UPPER):
        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)
        # small kernel for morph open - pre-allocated
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, bgr_small):
        """
        bgr_small: BGR image at detect resolution
        returns: (x,y) or None
        """
        # In-place friendly: convert to HSV (reuses memory)
        hsv = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        # Morph open to remove noise - cheap
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        # Find largest contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA * (bgr_small.shape[0] * bgr_small.shape[1]) / (320*240):
            return None
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)


class HybridTracker:
    """
    Unified interface with frame-skipping, scaling, and smoothing.
    Supports face (fill-light), hand (orb), color marker.
    Modes: face | mediapipe(hand) | color | auto (face -> hand -> color)
    Face returns (x,y,w,h, center); hand/color returns (x,y)
    For uniform API, face update returns dict with bbox+center.
    """
    def __init__(self, cfg=CFG, backend="auto", smoother_alpha=0.6, smoother_beta=0.15, face_detector="auto"):
        self.cfg = cfg
        self.backend_name = backend
        self.face_detector_name = face_detector
        # Use config FACE_SMOOTHER_ALPHA for zero lag (0.90 = instant)
        from config import FACE_SMOOTHER_ALPHA
        self.smoother = Smoother(alpha=smoother_alpha, beta=smoother_beta)
        self.face_smoother = Smoother(alpha=FACE_SMOOTHER_ALPHA, beta=0.30)
        self.det_w = cfg["detect_width"]
        self.det_h = cfg["detect_height"]
        self.cam_w = cfg["camera_width"]
        self.cam_h = cfg["camera_height"]
        self.scale_x = self.cam_w / self.det_w
        self.scale_y = self.cam_h / self.det_h

        self.tracker = None
        self.face_tracker = None
        self.head_pose_tracker = None
        if ENABLE_HEAD_POSE:
            try:
                self.head_pose_tracker = FaceDirectionTracker()
                print("[FaceDirection] MediaPipe Face Mesh ON - calibrated, confirmed direction commands")
            except Exception as e:
                # Keep control fail-safe if landmark tracking cannot start.
                print(f"[FaceDirection] Disabled ({e}); direction command remains STOP")
        self.gaze_tracker = None  # iris tracker removed - face bbox direction only
        self._init_backend()

        self.frame_count = 0
        self.last_pt = None
        self.last_face = None
        self.last_detect_time = 0
        self.last_face_center = None
        self.last_gaze = {"vector": (0, 0), "direction": "CENTER", "eyes": []}
        self.last_head_pose = {"vector": (0, 0), "direction": "NO FACE", "yaw": 0, "pitch": 0, "command": "STOP", "calibrated": False}

    def _init_backend(self):
        backend = self.backend_name
        if backend == "auto":
            # Prefer face for low-light fill use-case
            backend = "face"

        # Init primary tracker
        if backend == "face":
            # Haar is default - instant, no model download
            try:
                fd = self.face_detector_name
                if fd == "mediapipe" and MP_AVAILABLE:
                    self.face_tracker = MediaPipeFaceTracker()
                    self.face_detector_name = "mediapipe"
                elif fd == "haar" or fd == "auto":
                    self.face_tracker = FaceTrackerHaar()
                    self.face_detector_name = "haar"
                else:
                    self.face_tracker = FaceTrackerHaar()
                    self.face_detector_name = "haar"
                print(f"[Tracker] Face backend ({self.face_detector_name} ~2ms)")
            except Exception as e:
                print(f"[Tracker] Face init failed ({e}), fallback to Haar")
                self.face_tracker = FaceTrackerHaar()
                self.face_detector_name = "haar"
            # Also init hand tracker as secondary for auto
            if MP_AVAILABLE:
                try:
                    self.tracker = MediaPipeTracker(complexity=self.cfg.get("mediapipe_complexity", 0))
                except:
                    self.tracker = ColorTracker()
            else:
                self.tracker = ColorTracker()
            self.backend_name = "face"
        elif backend == "mediapipe":
            if not MP_AVAILABLE:
                print("[Tracker] mediapipe not available, falling back to color tracker")
                self.tracker = ColorTracker()
                self.backend_name = "color"
            else:
                try:
                    self.tracker = MediaPipeTracker(complexity=self.cfg.get("mediapipe_complexity", 0))
                    self.backend_name = "mediapipe"
                    print(f"[Tracker] MediaPipe Hand backend (complexity={self.cfg.get('mediapipe_complexity',0)})")
                except Exception as e:
                    print(f"[Tracker] MediaPipe init failed ({e}), fallback to color")
                    self.tracker = ColorTracker()
                    self.backend_name = "color"
        elif backend == "color":
            self.tracker = ColorTracker()
            self.backend_name = "color"
            print("[Tracker] Color (HSV) backend")
        else:
            raise ValueError(f"Unknown backend {backend}")

    def _detect_face(self, frame_bgr):
        """Returns face bbox (x,y,w,h) in detect space or None"""
        if frame_bgr.shape[1] == self.det_w and frame_bgr.shape[0] == self.det_h:
            small = frame_bgr
        else:
            small = cv2.resize(frame_bgr, (self.det_w, self.det_h), interpolation=cv2.INTER_LINEAR)
        if self.face_detector_name == "mediapipe":
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            return self.face_tracker.detect(rgb)
        else:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            return self.face_tracker.detect(gray)

    def update(self, frame_bgr):
        """
        frame_bgr: full-res BGR frame (cam_w x cam_h)
        Returns: dict with keys:
          'face': (x,y,w,h) in cam space or None
          'face_center': (cx,cy) smoothed or None
          'hand': (x,y) smoothed or None
          'low_light': bool
        For backward compat, also returns hand point via .last_pt but new code should use dict.
        """
        self.frame_count += 1
        n = self.cfg.get("detect_every_n_frames", 1)

        # On skipped frames, return last known (interpolated)
        if (self.frame_count % n) != 0 and (self.last_face is not None or self.last_pt is not None):
            # Smooth interpolate
            fc = self.face_smoother.update(self.last_face_center) if self.last_face_center else None
            hp = self.smoother.update(self.last_pt) if self.last_pt else None
            return {"face": self.last_face, "face_center": fc, "hand": hp, "low_light": False}

        t0 = time.perf_counter()
        face_small = None
        hand_small = None

        # Always try face first (if face backend or auto)
        if self.backend_name == "face":
            face_small = self._detect_face(frame_bgr)
            # Also try hand as secondary if face not found (auto behavior)
            if face_small is None and self.tracker is not None:
                # hand detect
                if frame_bgr.shape[1] == self.det_w and frame_bgr.shape[0] == self.det_h:
                    small = frame_bgr
                else:
                    small = cv2.resize(frame_bgr, (self.det_w, self.det_h), interpolation=cv2.INTER_LINEAR)
                if isinstance(self.tracker, MediaPipeTracker):
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    hand_small = self.tracker.detect(rgb)
                else:
                    hand_small = self.tracker.detect(small)
        elif self.backend_name == "mediapipe":
            if frame_bgr.shape[1] == self.det_w and frame_bgr.shape[0] == self.det_h:
                small = frame_bgr
            else:
                small = cv2.resize(frame_bgr, (self.det_w, self.det_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            hand_small = self.tracker.detect(rgb)
        else:  # color
            if frame_bgr.shape[1] == self.det_w and frame_bgr.shape[0] == self.det_h:
                small = frame_bgr
            else:
                small = cv2.resize(frame_bgr, (self.det_w, self.det_h), interpolation=cv2.INTER_LINEAR)
            hand_small = self.tracker.detect(small)

        self.last_detect_time = (time.perf_counter() - t0) * 1000

        # Map face to cam space
        face_cam = None
        face_center = None
        if face_small is not None:
            x, y, w, h = face_small
            x_cam = int(x * self.scale_x)
            y_cam = int(y * self.scale_y)
            w_cam = int(w * self.scale_x)
            h_cam = int(h * self.scale_y)
            # clamp
            x_cam = max(0, min(self.cam_w - 1, x_cam))
            y_cam = max(0, min(self.cam_h - 1, y_cam))
            w_cam = max(10, min(self.cam_w - x_cam, w_cam))
            h_cam = max(10, min(self.cam_h - y_cam, h_cam))
            face_cam = (x_cam, y_cam, w_cam, h_cam)
            self.last_face = face_cam
            cx = x_cam + w_cam // 2
            cy = y_cam + h_cam // 2
            self.last_face_center = (cx, cy)
            face_center = self.face_smoother.update(self.last_face_center)
        else:
            # keep last face for 8 frames to avoid flicker, then clear
            if self.last_face and self.frame_count % 8 == 0:
                # decay - do not clear immediately
                pass

        # Landmark head pose is independent of face-box movement.
        gaze = self.last_gaze  # kept for compat, always CENTER now
        head_pose = self.last_head_pose
        if self.head_pose_tracker is not None:
            head_pose = self.head_pose_tracker.detect(frame_bgr)
            self.last_head_pose = head_pose
        elif ENABLE_HEAD_POSE:
            head_pose = {"vector": (0, 0), "direction": "UNAVAILABLE", "yaw": 0, "pitch": 0, "command": "STOP", "calibrated": False}
        # Primary is head_pose (face direction) - no gaze fallback
        primary_vec = head_pose["vector"]
        primary_dir = head_pose["direction"]
        command = head_pose.get("command", "STOP")

        hand_cam = None
        if hand_small is not None:
            x_cam = int(hand_small[0] * self.scale_x)
            y_cam = int(hand_small[1] * self.scale_y)
            x_cam = max(0, min(self.cam_w - 1, x_cam))
            y_cam = max(0, min(self.cam_h - 1, y_cam))
            self.last_pt = (x_cam, y_cam)
            hand_cam = self.smoother.update(self.last_pt)

        # For face mode, return face even if hand found; hand is secondary
        if self.backend_name == "face":
            return {"face": face_cam if face_cam is not None else self.last_face if self.frame_count % 6 != 0 else None,
                    "face_center": face_center if face_center is not None else (self.face_smoother.update(self.last_face_center) if self.last_face_center and face_cam is None else None),
                    "hand": hand_cam,
                    "gaze": gaze,
                    "head_pose": head_pose,
                    "primary_vec": primary_vec,
                    "primary_dir": primary_dir,
                    "command": command,
                    "low_light": False}

        # Legacy hand-only
        if hand_cam is None:
            return None
        return hand_cam

    def update_legacy(self, frame_bgr):
        """Backward compat: returns (x,y) hand point or None"""
        res = self.update(frame_bgr)
        if isinstance(res, dict):
            return res.get("hand") or res.get("face_center")
        return res

    def get_last_detect_latency_ms(self):
        return self.last_detect_time

    def reset_calibration(self):
        """Manual recalibrate face CENTER - user pressed 'r'"""
        if self.head_pose_tracker:
            self.head_pose_tracker.reset_calibration()

    def close(self):
        if hasattr(self.tracker, 'close'):
            try: self.tracker.close()
            except: pass
        if self.face_tracker and hasattr(self.face_tracker, 'close'):
            try: self.face_tracker.close()
            except: pass
        if hasattr(self, 'head_pose_tracker') and self.head_pose_tracker and hasattr(self.head_pose_tracker, 'close'):
            try: self.head_pose_tracker.close()
            except: pass
