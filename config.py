"""
config.py - Central configuration for Virtual Light System
Optimized profiles for Pi 4 vs Pi 5 / fallback
"""
import cv2

# --- Profile Definitions --- Optimized for Pi5 / Pi4 / Pi4 1GB RAM / PC debug
PROFILES = {
    "pi5": {
        "camera_width": 640,
        "camera_height": 480,
        "detect_width": 256,
        "detect_height": 192,
        "fps_target": 30,
        "detect_every_n_frames": 1,
        "mediapipe_complexity": 0,
        "face_mesh_every_n": 1,  # PERFECT: run every frame for reliable LEFT/RIGHT
        "light_radius": 110,
        "use_picamera2": True,
    },
    "pi4": {
        "camera_width": 640,
        "camera_height": 480,
        "detect_width": 256,
        "detect_height": 192,
        "fps_target": 24,
        "detect_every_n_frames": 1,
        "mediapipe_complexity": 0,
        "face_mesh_every_n": 1,
        "light_radius": 115,
        "use_picamera2": True,
    },
    "pi4_low": {
        "camera_width": 480,
        "camera_height": 360,
        "detect_width": 192,
        "detect_height": 144,
        "fps_target": 20,
        "detect_every_n_frames": 1,
        "mediapipe_complexity": 0,
        "light_radius": 100,
        "use_picamera2": False,
    },
    "pi4_1gb": {
        "camera_width": 480,
        "camera_height": 360,
        "detect_width": 192,
        "detect_height": 144,
        "fps_target": 25,
        "detect_every_n_frames": 2,  # Process every 2nd frame for 0-lag on 1GB RAM
        "mediapipe_complexity": 0,
        "face_mesh_every_n": 2,
        "light_radius": 80,
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
        "light_radius": 140,
        "use_picamera2": False,
    }
}

# --- Active Profile Selection ---
# Auto-detect: try to read /proc/cpuinfo, fallback to pc_debug
ACTIVE_PROFILE = "pi5"  # change to "pi4", "pi4_low", "pi4_1gb", or "pc_debug" as needed

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

# Serial Controller settings (Python <-> ESP32 motor controller)
ENABLE_SERIAL = True
SERIAL_PORT = "COM13"     # ESP32 COM Port on Windows (e.g. COM13) or /dev/ttyUSB0 on Pi/Linux
SERIAL_BAUDRATE = 115200

# Smoothing: ZERO LAG - instant follow
SMOOTHING_ALPHA = 0.88  # was 0.6 -> 0.88 instant (higher = less lag)
SMOOTHING_BETA = 0.35  # was 0.15 -> faster when moving

# Light rendering - NATURAL SOFT LIGHT
LIGHT_COLOR_BGR = (255, 255, 255)
LIGHT_INTENSITY = 0.85
LIGHT_BLEND_MODE = "additive"
FACE_LIGHT_COLOR_BGR = (255, 255, 255)
FACE_LIGHT_INTENSITY = 0.85

# Detection thresholds
MEDIAPIPE_DETECTION_CONF = 0.5
MEDIAPIPE_TRACKING_CONF = 0.5

# Face detection (Haar - no extra model, ~2ms; MediaPipe Face ~5ms)
FACE_DETECTOR = "haar"  # "haar" | "mediapipe" | "auto"  (haar is fastest on Pi)
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE_RATIO = 0.12  # face must be at least 12% of frame height

# Low-light handling - BALANCED (no glare / no overexposure)
LOW_LIGHT_THRESHOLD = 80    # only trigger when truly dark
LOW_LIGHT_BOOST = True
AUTO_GLOW_BOOST_FACTOR = 1.25 # subtle boost, prevents blowing out bright background
FACE_GLOW_SCALE = 1.5       # natural size
AMBIENT_FILL_ALPHA = 0.05   # minimal global fill
GLOBAL_BRIGHTNESS_GAIN = 0  # set 0 to prevent background glare/whiteout

# Face tracking smoothing - ZERO LAG
FACE_SMOOTHER_ALPHA = 0.90  # was 0.5 -> instant follow, no delay

# Smart-wheelchair head-direction control
# MediaPipe Face Mesh + solvePnP measures head rotation from facial landmarks.
# This is deliberately less sensitive than the former face-box movement heuristic:
# a command is emitted only after it is stable for several frames.
ENABLE_HEAD_POSE = True
ENABLE_GAZE = False      # deprecated - iris tracker removed, keep False
PATIENT_MODE = True      # True = high sensitivity for patients with limited head movement
GAZE_HISTORY = 4

# Debug / Performance switch
DEBUG_MODE = True
PERFORMANCE_MODE = True

# Calibration tunables
FACE_AUTO_CALIBRATE = True
FACE_CALIBRATE_FRAMES = 8    # Fast 8 frames (~0.25s) for instant auto-calibration
FACE_CALIBRATE_SMOOTHING = 0.003
FACE_CALIB_YAW_TOL = 25.0
FACE_CALIB_PITCH_TOL = 25.0
FACE_CALIB_ROLL_TOL = 25.0
FACE_CALIB_FALLBACK_SEC = 1.0  # Rapid fallback after 1.0s
FACE_CALIB_TIMEOUT_SEC = 2.0   # Timeout 2s
FACE_CALIB_MIN_VALID = 4
FACE_CALIB_DEBUG = True
FACE_CALIB_CONF_THRESH = 0.4

# Responsive direction thresholds (gentle patient head movements emit commands)
HEAD_YAW_THRESHOLD = 5.5       # ENTER threshold for LEFT / RIGHT (degrees/units) - hyper sensitive & reachable!
HEAD_PITCH_THRESHOLD = 6.0     # ENTER threshold for FORWARD / BACKWARD (degrees/units)
HEAD_YAW_EXIT_THRESHOLD = 2.5  # EXIT threshold for LEFT / RIGHT
HEAD_PITCH_EXIT_THRESHOLD = 3.0# EXIT threshold for FORWARD / BACKWARD
HEAD_DIRECTION_HISTORY = 2     # 2 samples for immediate response
HEAD_SMOOTH_ALPHA = 0.70       # fast EMA filtering
COMMAND_CONFIRM_FRAMES = 2     # 2 frames confirmation
COMMAND_STOP_CONFIRM_FRAMES = 2
HEAD_MISSING_TOLERANCE = 8     # keep last direction for N frames when face briefly lost

HEAD_DIRECTION_DEBUG = True
DEBUG_DIRECTION = True
HEAD_DIRECTION_INVERT_X = False
HEAD_DIRECTION_INVERT_Y = False
BODY_DIRECTION_INVERT = False
DIRECTION_CONFIRM_FRAMES = 2

# Body pose disabled by default for 30 FPS single-pass FaceMesh performance
ENABLE_BODY_POSE = False
BODY_YAW_ENTER_THRESHOLD = 8.0
BODY_YAW_EXIT_THRESHOLD = 4.0
BODY_PITCH_ENTER_THRESHOLD = 8.0
BODY_PITCH_EXIT_THRESHOLD = 4.0
BODY_SHOULDER_OFFSET_ENTER = 0.018
BODY_SHOULDER_OFFSET_EXIT = 0.010
BODY_POSE_CONF_THRESH = 0.4
BODY_DIRECTION_HISTORY = 3
BODY_SMOOTH_ALPHA = 0.50
BODY_COMMAND_CONFIRM_FRAMES = 2
BODY_MISSING_TOLERANCE = 6
BODY_CALIB_FRAMES = 10
BODY_CALIB_TIMEOUT_SEC = 3.0
HEAD_SENSITIVITY = 95

# Tune via tools/calibrate_color.py
HSV_LOWER = (40, 70, 70)
HSV_UPPER = (80, 255, 255)
MIN_CONTOUR_AREA = 400

# Performance
ENABLE_FPS_DISPLAY = True
ENABLE_LATENCY_DISPLAY = True
V4L2_FOURCC = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG is faster than YUYV on Pi
CAMERA_FPS_REQUEST = 30
