import cv2
import numpy as np
import time
import mediapipe as mp

print("Testing MediaPipe FaceMesh performance...")
face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Test frame 640x480 vs 320x240 vs 256x192
for w, h in [(640, 480), (320, 240), (256, 192)]:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.circle(img, (w//2, h//2), 50, (255, 255, 255), -1)
    
    # Warmup
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_mesh.process(rgb)
    
    t0 = time.perf_counter()
    N = 30
    for _ in range(N):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
    dt = (time.perf_counter() - t0) * 1000 / N
    print(f"Size {w}x{h}: {dt:.2f} ms per frame -> Potential FPS: {1000/dt:.1f}")

face_mesh.close()
