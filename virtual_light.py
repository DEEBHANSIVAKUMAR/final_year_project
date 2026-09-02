"""
virtual_light.py - Precomputed glow texture + ultra-fast blending + FACE fill-light
Key optimization: Gaussian blur is precomputed ONCE at startup, not per-frame.
Per-frame cost is just ROI copy + alpha/additive blend (~0.5-1.5ms at 640x480).
Face mode: large soft glow centered on face + global low-light fill.
"""
import cv2
import numpy as np


class VirtualLight:
    def __init__(self, radius=80, color_bgr=(0, 220, 255), intensity=0.85, mode="additive"):
        """
        radius: glow radius in pixels
        color_bgr: BGR tuple
        intensity: 0..1
        mode: "additive" (brighter, realistic) or "alpha" (transparent overlay)
        """
        self.radius = radius
        self.color = np.array(color_bgr, dtype=np.uint8)
        self.intensity = intensity
        self.mode = mode
        self.size = radius * 2
        # Precompute textures
        self.texture_bgr, self.alpha = self._build_texture(radius, color_bgr, intensity)
        # For additive: precompute float texture scaled
        self.texture_f = self.texture_bgr.astype(np.float32)
        # Cache for scaled face glows: radius -> (tex, alpha)
        self._cache = {}
        # Reusable overlay for ambient fill (avoid alloc per frame)
        self._overlay = None

    def _build_texture(self, radius, color_bgr, intensity):
        """
        Build radial gradient glow: center bright -> edges fade
        Uses vectorized numpy, single Gaussian blur at init only.
        """
        size = radius * 2
        # Create single-channel radial falloff
        y, x = np.ogrid[:size, :size]
        cx, cy = radius, radius
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        # Normalize 0..1
        norm = np.clip(1 - dist / radius, 0, 1)
        # Smooth falloff (squared for more realistic glow) + small blur
        falloff = (norm ** 1.8)  # exponent controls softness
        # Apply Gaussian blur ONCE
        falloff_uint8 = (falloff * 255).astype(np.uint8)
        falloff_uint8 = cv2.GaussianBlur(falloff_uint8, (0, 0), sigmaX=radius * 0.35)

        # Alpha channel from blurred falloff
        alpha = (falloff_uint8.astype(np.float32) / 255.0) * intensity  # 0..intensity
        alpha_3ch = cv2.merge([alpha, alpha, alpha])  # HxWx3 float32 0..1

        # BGR texture: color * falloff
        color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
        # Use blurred falloff as intensity map
        intensity_map = falloff_uint8.astype(np.float32) / 255.0  # 0..1
        intensity_map_3ch = np.stack([intensity_map]*3, axis=-1)
        texture = (color * intensity_map_3ch).astype(np.uint8)

        return texture, alpha_3ch.astype(np.float32)

    def render(self, frame, center):
        """
        Blend glow onto frame IN-PLACE for zero extra copy.
        frame: BGR uint8 (H x W x 3)
        center: (x, y) or None
        Returns frame (modified in-place)
        """
        if center is None:
            return frame

        x, y = center
        h, w = frame.shape[:2]
        r = self.radius
        size = self.size

        # Compute ROI bounds with clipping
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(w, x + r)
        y2 = min(h, y + r)

        if x1 >= x2 or y1 >= y2:
            return frame

        # Texture ROI (if clipped at edges)
        tx1 = r - (x - x1)
        ty1 = r - (y - y1)
        tx2 = tx1 + (x2 - x1)
        ty2 = ty1 + (y2 - y1)

        roi = frame[y1:y2, x1:x2]  # view, no copy
        tex = self.texture_bgr[ty1:ty2, tx1:tx2]
        alp = self.alpha[ty1:ty2, tx1:tx2]

        if self.mode == "additive":
            # Additive: roi * (1 - alpha) + texture  -> but brighter
            # Faster: roi = roi + texture*alpha  (using cv2.add)
            # Vectorized blend: roi = roi*(1 - alp) + tex*alp  -> alpha blend
            # For additive realism, do: roi = cv2.add(roi, (tex*alp).astype(uint8))
            blended = (tex.astype(np.float32) * alp).astype(np.uint8)
            # cv2.add handles saturation at 255 (no wrap)
            np.copyto(roi, cv2.add(roi, blended))
        else:
            # Standard alpha blend
            # roi = roi*(1 - alp) + tex*alp
            # Using float32 then convert
            roi_f = roi.astype(np.float32)
            tex_f = tex.astype(np.float32)
            # This is ~0.8ms for 160x160 ROI
            blended_f = roi_f * (1.0 - alp) + tex_f * alp
            np.copyto(roi, blended_f.astype(np.uint8))

        # Optional inner core (small bright circle) for realistic bulb
        # Draw directly on frame, very cheap
        cv2.circle(frame, (x, y), max(3, r // 10), (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x, y), max(6, r // 6), tuple(int(c) for c in self.color), -1, lineType=cv2.LINE_AA)

        return frame

    def update_color(self, color_bgr):
        """Hot-swap color without rebuilding blur kernel from scratch"""
        self.color = np.array(color_bgr, dtype=np.uint8)
        self.texture_bgr, self.alpha = self._build_texture(self.radius, color_bgr, self.intensity)

    # --- Face / Low-light extensions ---

    def render_face(self, frame, face_bbox, face_center=None, brightness=1.0, low_light=False):
        """
        Face-triggered virtual fill light.
        face_bbox: (x,y,w,h) in frame coords or None
        face_center: (cx,cy) or derived from bbox
        brightness: 0..255 mean V of frame (for auto boost)
        low_light: bool (mean V < threshold)
        Returns frame in-place.
        Effect:
          1) Large soft glow centered on face -> like ring light
          2) In low light: fast ROI brighten + optional ambient (only when no face)
        Cost: ~2-5ms (cached texture, fast BGR add)
        """
        if face_bbox is None and face_center is None:
            return frame

        if face_bbox is not None:
            x, y, w, h = face_bbox
            cx = x + w // 2
            cy = y + h // 2
            # radius BIGGER - 0.95x face + base radius (user wants big circle)
            r = int(max(w, h) * 0.95) + int(self.radius * 0.35)
            r = max(r, self.radius)
            r = min(r, 260)  # was 180 -> allow much bigger
        elif face_center is not None:
            cx, cy = face_center
            r = self.radius
            x = y = w = h = None
        else:
            return frame

        try:
            from config import AUTO_GLOW_BOOST_FACTOR
            boost = AUTO_GLOW_BOOST_FACTOR
        except:
            boost = 3.0
        intensity_boost = boost if low_light else boost*0.55
        self._blend_at(frame, (cx, cy), radius=r, intensity_scale=intensity_boost)
        # Second pass core - even brighter white
        self._blend_at(frame, (cx, cy), radius=int(r*0.58), intensity_scale=boost*0.62)
        # Third tiny core for pure white highlight
        if low_light:
            self._blend_at(frame, (cx, cy), radius=int(r*0.32), intensity_scale=1.0)

        if low_light and face_bbox is not None:
            try:
                from config import AMBIENT_FILL_ALPHA
                if AMBIENT_FILL_ALPHA > 0:
                    self._ambient_fill(frame, alpha=AMBIENT_FILL_ALPHA * 0.5)
            except:
                pass

        # Subtle ring to indicate virtual light active (pure white)
        if face_bbox is not None:
            cv2.ellipse(frame, (cx, cy), (w//2 + 6, h//2 + 10), 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _blend_at(self, frame, center, radius=None, intensity_scale=1.0):
        """Blend precomputed glow at arbitrary radius (scales cached texture if needed)"""
        if center is None:
            return frame
        x, y = center
        r = radius if radius is not None else self.radius
        # Cache lookup for scaled radii (avoid resize per frame)
        if r == self.radius:
            tex = self.texture_bgr
            alp = self.alpha
            if intensity_scale != 1.0:
                alp = np.clip(self.alpha * intensity_scale, 0, 1.0)
        else:
            key = (r, round(intensity_scale, 2))
            cached = self._cache.get(key)
            if cached is not None:
                tex, alp = cached
            else:
                new_size = r * 2
                tex = cv2.resize(self.texture_bgr, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
                base_alpha = cv2.resize(self.alpha, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
                alp = np.clip(base_alpha * np.clip(intensity_scale, 0.5, 2.0), 0, 1.0)
                # Evict if cache too large (keep last 8)
                if len(self._cache) > 8:
                    self._cache.pop(next(iter(self._cache)))
                self._cache[key] = (tex, alp)

        h, w = frame.shape[:2]
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(w, x + r)
        y2 = min(h, y + r)
        if x1 >= x2 or y1 >= y2:
            return frame
        tx1 = r - (x - x1)
        ty1 = r - (y - y1)
        tx2 = tx1 + (x2 - x1)
        ty2 = ty1 + (y2 - y1)
        roi = frame[y1:y2, x1:x2]
        t = tex[ty1:ty2, tx1:tx2]
        a = alp[ty1:ty2, tx1:tx2]
        if self.mode == "additive":
            blended = (t.astype(np.float32) * a).astype(np.uint8)
            np.copyto(roi, cv2.add(roi, blended))
        else:
            roi_f = roi.astype(np.float32)
            tex_f = t.astype(np.float32)
            blended_f = roi_f * (1.0 - a) + tex_f * a
            np.copyto(roi, blended_f.astype(np.uint8))
        # inner core
        cv2.circle(frame, (x, y), max(3, r // 12), (255, 255, 255), -1, lineType=cv2.LINE_AA)
        return frame

    def _ambient_fill(self, frame, alpha=0.12):
        """Cheap global fill: warm wash via in-place weighted add (no alloc if possible)"""
        if alpha <= 0:
            return frame
        alpha = float(np.clip(alpha, 0, 0.3))
        # Reuse overlay buffer to avoid per-frame alloc (0.9MB)
        h, w = frame.shape[:2]
        if self._overlay is None or self._overlay.shape[:2] != (h, w):
            self._overlay = np.full((h, w, 3), self.color, dtype=np.uint8)
        # Fast: frame = frame*(1-a) + overlay*a
        cv2.addWeighted(self._overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    @staticmethod
    def estimate_brightness(frame):
        """Fast low-light estimate: mean V in HSV, sampled 1/4 res (~0.2ms)"""
        h, w = frame.shape[:2]
        small = frame[::4, ::4]  # 1/16 pixels
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        return float(np.mean(hsv[:, :, 2]))
