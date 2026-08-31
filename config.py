"""
config.py - Central configuration for Virtual Light System
Optimized profiles for Pi 4 vs Pi 5 / fallback
"""
import cv2

# --- Profile Definitions --- BIGGER CIRCLE
PROFILES = {
    "pi5": {
        "camera_width": 640,
        "camera_height": 480,
        "detect_width": 320,
        "detect_height": 240,
        "fps_target": 30,
        "detect_every_n_frames": 1,
        "mediapipe_complexity": 0,
        "light_radius": 130,  # was 80 -> bigger circle
        "use_picamera2": True,
    },
    "pi4": {
        "camera_width": 640,
        "camera_height": 480,
        "detect_width": 256,
        "detect_height": 192,
        "fps_target": 24,
        "detect_every_n_frames": 1,  # was 2 -> zero lag (process every frame)
        "mediapipe_complexity": 0,
        "light_radius": 115,  # was 60
        "use_picamera2": True,
    },
    "pi4_low": {
        "camera_width": 480,
        "camera_height": 360,
        "detect_width": 192,
        "detect_height": 144,
        "fps_target": 20,
        "detect_every_n_frames": 1,  # was 3 -> faster response
        "mediapipe_complexity": 0,
        "light_radius": 100,  # was 50
        "use_picamera2": False,
    },
    "pc_debug": {
        "camera_width": 640,
        "camera_height": 480,
        "detect_width": 320,
        "detect_height": 240,
        "fps_target": 30,
        "detect_every_n_frames": 1,
        "mediapipe_complexity": 0,
        "light_radius": 140,  # was 80 -> BIG circle for PC demo
        "use_picamera2": False,
    }
}

# --- Active Profile Selection ---
# Auto-detect: try to read /proc/cpuinfo, fallback to pc_debug
ACTIVE_PROFILE = "pi5"  # change to "pi4", "pi4_low", or "pc_debug" as needed

def get_config(profile: str = None):
    name = profile or ACTIVE_PROFILE
    cfg = PROFILES.get(name, PROFILES["pi5"]).copy()
    cfg["profile_name"] = name
    return cfg

# --- Global Tunables (shared across profiles) ---
CFG = get_config()

# Tracker settings
# "face" = face-triggered fill light (recommended for low-light perfect lighting)
# "mediapipe" = hand fingertip, "color" = HSV marker, "auto" = face -> hand -> color
TRACKER_BACKEND = "face"  # "face" | "mediapipe" | "color" | "auto"
TRACKER_MODE = "face"     # alias for main.py --mode

# Smoothing: ZERO LAG - instant follow
SMOOTHING_ALPHA = 0.88  # was 0.6 -> 0.88 instant (higher = less lag)
SMOOTHING_BETA = 0.35  # was 0.15 -> faster when moving

# Light rendering - PURE WHITE ULTRA BRIGHT
LIGHT_COLOR_BGR = (255, 255, 255)
LIGHT_INTENSITY = 1.0  # was 0.98 -> max
LIGHT_BLEND_MODE = "additive"
FACE_LIGHT_COLOR_BGR = (255, 255, 255)
FACE_LIGHT_INTENSITY = 1.0  # was 0.95 -> max

# Detection thresholds
MEDIAPIPE_DETECTION_CONF = 0.5
MEDIAPIPE_TRACKING_CONF = 0.5

# Face detection (Haar - no extra model, ~2ms; MediaPipe Face ~5ms)
FACE_DETECTOR = "haar"  # "haar" | "mediapipe" | "auto"  (haar is fastest on Pi)
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE_RATIO = 0.12  # face must be at least 12% of frame height

# Low-light handling - MAX BRIGHT (user wants brighter)
LOW_LIGHT_THRESHOLD = 125   # was 110 -> brighter even in normal light
LOW_LIGHT_BOOST = True
AUTO_GLOW_BOOST_FACTOR = 3.0  # was 2.4 -> much stronger
FACE_GLOW_SCALE = 2.4      # was 2.1 -> bigger halo
AMBIENT_FILL_ALPHA = 0.38  # was 0.32 -> max global fill
GLOBAL_BRIGHTNESS_GAIN = 22  # extra BGR add in low light (0-40)

# Face tracking smoothing - ZERO LAG
FACE_SMOOTHER_ALPHA = 0.90  # was 0.5 -> instant follow, no delay

# Smart-wheelchair head-direction control
# MediaPipe Face Mesh + solvePnP measures head rotation from facial landmarks.
# This is deliberately less sensitive than the former face-box movement heuristic:
# a command is emitted only after it is stable for several frames.
ENABLE_HEAD_POSE = True
ENABLE_GAZE = False      # deprecated - iris tracker removed, keep False
PATIENT_MODE = True      # True = high sensitivity for patients with limited head movement
GAZE_HISTORY = 4         # smoothing history (face uses PATIENT mode: 1 else 2)
# Keep the head straight while the initial calibration completes.
FACE_AUTO_CALIBRATE = True
FACE_CALIBRATE_FRAMES = 30
FACE_CALIBRATE_SMOOTHING = 0.003
# Relative angle thresholds after calibration.  Tune only after real-world tests.
HEAD_YAW_THRESHOLD = 15.0      # LEFT / RIGHT
HEAD_PITCH_THRESHOLD = 12.0    # FORWARD / BACKWARD
HEAD_DIRECTION_HISTORY = 5     # pose samples averaged to reduce jitter
COMMAND_CONFIRM_FRAMES = 8     # identical non-STOP frames required before a command
HEAD_DIRECTION_INVERT_X = False
HEAD_DIRECTION_INVERT_Y = False
# Retained for the on-screen visual-light direction indicator.
HEAD_SENSITIVITY = 95

# Color tracker HSV range (for fallback - e.g., bright green glove/marker)
# Tune via tools/calibrate_color.py
HSV_LOWER = (40, 70, 70)
HSV_UPPER = (80, 255, 255)
MIN_CONTOUR_AREA = 400

# Performance
ENABLE_FPS_DISPLAY = True
ENABLE_LATENCY_DISPLAY = True
V4L2_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG is faster than YUYV on Pi
CAMERA_FPS_REQUEST = 30
