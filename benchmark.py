"""
benchmark.py - Measure detector + render latency without camera
Runs synthetic frames to estimate FPS on current hardware.
Useful for Pi tuning before connecting camera.
"""
import cv2
import numpy as np
import time
from config import get_config
from tracker import HybridTracker
from virtual_light import VirtualLight

def bench(profile="pi5", backend="color", n=100):
    cfg = get_config(profile)
    print(f"Benchmark: profile={profile} backend={backend} {cfg['camera_width']}x{cfg['camera_height']} -> {cfg['detect_width']}x{cfg['detect_height']} skip={cfg['detect_every_n_frames']}")
    tracker = HybridTracker(cfg=cfg, backend=backend)
    light = VirtualLight(radius=cfg["light_radius"], color_bgr=(0,220,255), intensity=0.85, mode="additive")

    # Synthetic frame
    frame = np.random.randint(0, 255, (cfg["camera_height"], cfg["camera_width"], 3), dtype=np.uint8)
    # Add a green blob for color tracker
    cv2.circle(frame, (cfg["camera_width"]//2, cfg["camera_height"]//2), 30, (0,255,0), -1)

    times = []
    detect_times = []
    render_times = []

    for i in range(n):
        t0 = time.perf_counter()
        pt = tracker.update(frame)
        dt_detect = tracker.get_last_detect_latency_ms()
        # move fake blob
        cx = (cfg["camera_width"]//2 + int(40*np.sin(i*0.2))) % cfg["camera_width"]
        cy = (cfg["camera_height"]//2 + int(30*np.cos(i*0.15))) % cfg["camera_height"]
        cv2.circle(frame, (cx, cy), 20, (0,255,0), -1)

        t1 = time.perf_counter()
        if isinstance(pt, dict):
            pt = pt.get("face_center") or pt.get("hand")
        if pt is None:
            pt = (cx, cy)
        light.render(frame, pt)
        t2 = time.perf_counter()

        times.append((t2 - t0)*1000)
        detect_times.append(dt_detect)
        render_times.append((t2 - t1)*1000)

    tracker.close()
    avg = np.mean(times)
    print(f"Total per frame: avg {avg:.2f}ms | median {np.median(times):.2f}ms | p95 {np.percentile(times,95):.2f}ms")
    print(f"Detect: avg {np.mean(detect_times):.2f}ms | Render: avg {np.mean(render_times):.2f}ms")
    print(f"Estimated FPS (no display): {1000/avg:.1f}")
    print(f"Estimated FPS (with display ~2ms overhead): {1000/(avg+2):.1f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="pi5")
    p.add_argument("--backend", default="face")
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args()
    bench(args.profile, args.backend, args.n)
    # Also test mediapipe if available
    try:
        import mediapipe
        if args.backend != "mediapipe":
            print("\n--- Also testing mediapipe ---")
            bench(args.profile, "mediapipe", args.n)
    except ImportError:
        pass
