"""
tracker.py - PERFECT SIMPLIFIED TRACKER FOR WHEELCHAIR
Fixes: 0/25 stuck, STOP always, LEFT/RIGHT never
Primary: BODY (shoulders 11,12 hips 23,24) via normalized_turn = (right_z-left_z)/width
Fallback: FACE yaw via FaceMesh solvePnP
Clean, fast, patient-friendly, works on PC and Pi5
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

print(f"RUNNING FILE: {os.path.abspath(__file__)}")
print("VERSION: DIRECTION_DEBUG_V2")
import config as config_mod
from config import CFG, MEDIAPIPE_DETECTION_CONF, MEDIAPIPE_TRACKING_CONF, HSV_LOWER, HSV_UPPER, MIN_CONTOUR_AREA, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBORS, FACE_MIN_SIZE_RATIO, ENABLE_HEAD_POSE, HEAD_YAW_THRESHOLD, HEAD_PITCH_THRESHOLD, HEAD_YAW_EXIT_THRESHOLD, HEAD_PITCH_EXIT_THRESHOLD, HEAD_SMOOTH_ALPHA, FACE_AUTO_CALIBRATE, FACE_CALIBRATE_FRAMES, FACE_CALIBRATE_SMOOTHING, FACE_CALIB_YAW_TOL, FACE_CALIB_PITCH_TOL, FACE_CALIB_ROLL_TOL, FACE_CALIB_FALLBACK_SEC, FACE_CALIB_TIMEOUT_SEC, FACE_CALIB_MIN_VALID, FACE_CALIB_DEBUG, FACE_CALIB_CONF_THRESH, HEAD_DIRECTION_HISTORY, COMMAND_CONFIRM_FRAMES, COMMAND_STOP_CONFIRM_FRAMES, HEAD_MISSING_TOLERANCE, HEAD_DIRECTION_INVERT_X, HEAD_DIRECTION_INVERT_Y, DEBUG_MODE, PERFORMANCE_MODE, ENABLE_BODY_POSE, BODY_YAW_ENTER_THRESHOLD, BODY_YAW_EXIT_THRESHOLD, BODY_PITCH_ENTER_THRESHOLD, BODY_PITCH_EXIT_THRESHOLD, BODY_SHOULDER_OFFSET_ENTER, BODY_SHOULDER_OFFSET_EXIT, BODY_POSE_CONF_THRESH, BODY_DIRECTION_HISTORY, BODY_SMOOTH_ALPHA, BODY_COMMAND_CONFIRM_FRAMES, BODY_MISSING_TOLERANCE, BODY_CALIB_FRAMES, BODY_CALIB_TIMEOUT_SEC, DEBUG_DIRECTION, BODY_DIRECTION_INVERT, DIRECTION_CONFIRM_FRAMES

class Smoother:
    def __init__(self, alpha=0.6, beta=0.15):
        self.alpha=alpha; self.beta=beta; self.prev=None; self.velocity=np.zeros(2,dtype=np.float32)
    def update(self, pt):
        if pt is None: return None
        pt=np.array(pt,dtype=np.float32)
        if self.prev is None:
            self.prev=pt; return tuple(pt.astype(int))
        vel=pt-self.prev; self.velocity=0.3*vel+0.7*self.velocity
        speed=np.linalg.norm(self.velocity)
        a=np.clip(self.alpha+self.beta*(speed/50.0),0.2,0.95)
        smoothed=a*pt+(1-a)*self.prev; self.prev=smoothed
        return tuple(smoothed.astype(int))
    def reset(self):
        self.prev=None; self.velocity=np.zeros(2,dtype=np.float32)

class FaceTrackerHaar:
    def __init__(self):
        p=cv2.data.haarcascades+"haarcascade_frontalface_default.xml"
        if not os.path.exists(p):
            p=os.path.join(os.path.dirname(cv2.__file__),"data","haarcascade_frontalface_default.xml")
        self.face_cascade=cv2.CascadeClassifier(p)
        if self.face_cascade.empty(): raise RuntimeError(f"Haar load failed {p}")
        self.min_size_ratio=FACE_MIN_SIZE_RATIO
    def detect(self,gray_small):
        h,w=gray_small.shape[:2]
        min_h=int(h*self.min_size_ratio); min_w=int(w*self.min_size_ratio*0.8)
        eq=cv2.equalizeHist(gray_small)
        faces=self.face_cascade.detectMultiScale(eq, scaleFactor=FACE_SCALE_FACTOR, minNeighbors=FACE_MIN_NEIGHBORS, minSize=(min_w,min_h), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces)==0: return None
        return tuple(max(faces,key=lambda r:r[2]*r[3]))

# ---------- FACE POSE ----------
class FaceDirectionTracker:
    def __init__(self):
        if not MP_AVAILABLE: raise RuntimeError("MediaPipe required")
        conf=float(FACE_CALIB_CONF_THRESH)
        self.face_mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=False,max_num_faces=1,refine_landmarks=False,min_detection_confidence=conf,min_tracking_confidence=conf)
        self.history=[]; self.hist_len=HEAD_DIRECTION_HISTORY
        self.yaw_enter=float(HEAD_YAW_THRESHOLD); self.pitch_enter=float(HEAD_PITCH_THRESHOLD)
        self.yaw_exit=float(HEAD_YAW_EXIT_THRESHOLD); self.pitch_exit=float(HEAD_PITCH_EXIT_THRESHOLD)
        self.smooth_alpha=float(HEAD_SMOOTH_ALPHA)
        self.filtered_yaw=None; self.filtered_pitch=None
        self.last_vec=(0.0,0.0); self.last_dir="STOP"; self.stable_command="STOP"; self.pending_command="STOP"; self.pending_frames=0; self.missing_frames=0
        self.confirm_frames=int(COMMAND_CONFIRM_FRAMES); self.stop_confirm=int(COMMAND_STOP_CONFIRM_FRAMES); self.missing_tolerance=int(HEAD_MISSING_TOLERANCE)
        self.calib_yaw=None; self.calib_pitch=None; self.calib_roll=None; self.calib_buffer=[]; self.calibrated=False
        self.calib_frames=int(FACE_CALIBRATE_FRAMES); self.auto_calibrate=bool(FACE_AUTO_CALIBRATE)
        self.calib_yaw_tol=float(FACE_CALIB_YAW_TOL); self.calib_pitch_tol=float(FACE_CALIB_PITCH_TOL); self.calib_roll_tol=float(FACE_CALIB_ROLL_TOL)
        self.calib_fallback=float(FACE_CALIB_FALLBACK_SEC); self.calib_timeout=float(FACE_CALIB_TIMEOUT_SEC); self.calib_min_valid=int(FACE_CALIB_MIN_VALID)
        self.calib_start_time=None; self.calib_frame_count=0; self.calib_face_seen=0; self._perf_idx=0; self._cached_pose=None
    def reset_calibration(self):
        self.calib_yaw=self.calib_pitch=self.calib_roll=None; self.calib_buffer=[]; self.calibrated=False
        self.history.clear(); self.filtered_yaw=self.filtered_pitch=None
        self.last_vec=(0.0,0.0); self.last_dir="STOP"; self.stable_command="STOP"; self.pending_command="STOP"; self.pending_frames=0; self.missing_frames=0
        self.calib_start_time=None; self.calib_frame_count=0; self.calib_face_seen=0; self._cached_pose=None
        print("[FaceDirection] RESET - keep face straight")
    @staticmethod
    def _rotation_angles(rvec):
        rot,_=cv2.Rodrigues(rvec)
        sy=np.sqrt(rot[0,0]**2+rot[1,0]**2)
        pitch=np.degrees(np.arctan2(rot[2,1],rot[2,2]))
        yaw=np.degrees(np.arctan2(-rot[2,0],sy))
        roll=np.degrees(np.arctan2(rot[1,0],rot[0,0]))
        return float(yaw),float(pitch),float(roll)
    def _measure_pose(self,frame_bgr):
        h,w=frame_bgr.shape[:2]
        if w>320 or h>240:
            small=cv2.resize(frame_bgr,(320,240),interpolation=cv2.INTER_LINEAR)
            rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB); h,w=small.shape[:2]
        else:
            rgb=cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB)
        res=self.face_mesh.process(rgb)
        if not res.multi_face_landmarks: return None
        lm=res.multi_face_landmarks[0].landmark
        idxs=(1,152,33,263,61,291)
        img_pts=np.array([(lm[i].x*w,lm[i].y*h) for i in idxs],dtype=np.float64)
        mod_pts=np.array([(0,0,0),(0,-63.6,-12.5),(-43.3,32.7,-26.0),(43.3,32.7,-26.0),(-28.9,-28.9,-24.1),(28.9,-28.9,-24.1)],dtype=np.float64)
        cam=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=np.float64)
        ok,rvec,_=cv2.solvePnP(mod_pts,img_pts,cam,np.zeros((4,1)),flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: return None
        return self._rotation_angles(rvec)
    def _raw(self,ay,ap):
        if ay<=-self.yaw_enter: return "LEFT"
        if ay>=self.yaw_enter: return "RIGHT"
        if ap<=-self.pitch_enter: return "FORWARD"
        if ap>=self.pitch_enter: return "BACKWARD"
        return "STOP"
    def _hyst(self,ay,ap):
        sc=self.stable_command
        if sc=="LEFT" and ay<=-self.yaw_exit: return "LEFT"
        if sc=="RIGHT" and ay>=self.yaw_exit: return "RIGHT"
        if sc=="FORWARD" and ap<=-self.pitch_exit: return "FORWARD"
        if sc=="BACKWARD" and ap>=self.pitch_exit: return "BACKWARD"
        return self._raw(ay,ap)
    def _confirm(self,cand):
        need=self.stop_confirm if cand=="STOP" else self.confirm_frames
        if cand!= "STOP": need=self.confirm_frames
        if cand==self.pending_command: self.pending_frames+=1
        else: self.pending_command=cand; self.pending_frames=1
        if self.pending_frames>=need: self.stable_command=cand
        return self.stable_command
    def _valid(self,yaw,pitch,roll):
        if abs(yaw)>self.calib_yaw_tol: return False,f"yaw {yaw:.1f}>{self.calib_yaw_tol:.0f}"
        if abs(pitch)>self.calib_pitch_tol: return False,f"pitch {pitch:.1f}>{self.calib_pitch_tol:.0f}"
        if abs(roll)>self.calib_roll_tol: return False,f"roll {roll:.1f}>{self.calib_roll_tol:.0f}"
        return True,""
    def _finalize(self,reason=""):
        if len(self.calib_buffer)>=1:
            self.calib_yaw=float(np.median([p[0] for p in self.calib_buffer]))
            self.calib_pitch=float(np.median([p[1] for p in self.calib_buffer]))
            print(f"[FaceDirection] CALIBRATED yaw={self.calib_yaw:.1f} pitch={self.calib_pitch:.1f} ({len(self.calib_buffer)}/{self.calib_frames}) {reason}")
        else:
            self.calib_yaw=self.calib_pitch=0.0
            print(f"[FaceDirection] Fallback 0,0 {reason}")
        self.calibrated=True; self.history.clear(); self.filtered_yaw=self.filtered_pitch=None
    def detect(self,frame_bgr,face_bbox=None):
        if not ENABLE_HEAD_POSE:
            return {"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"roll":0,"calibrated":False,"command":"STOP","face_detected":False}
        if self.auto_calibrate and not self.calibrated and self.calib_start_time is None:
            self.calib_start_time=time.time(); self.calib_face_seen=0; self.calib_frame_count=0
        self._perf_idx+=1
        use_cached=False
        if self.calibrated and PERFORMANCE_MODE and self._cached_pose is not None and self.missing_frames==0:
            n=CFG.get("face_mesh_every_n",2)
            if n>1 and (self._perf_idx % n)!=0: use_cached=True
        pose=self._cached_pose if use_cached else self._measure_pose(frame_bgr)
        if not use_cached:
            if pose is not None: self._cached_pose=pose
        face_has_bbox=face_bbox is not None
        # --- calibration ---
        if self.auto_calibrate and not self.calibrated:
            self.calib_frame_count+=1
            if pose is not None or face_has_bbox: self.calib_face_seen+=1
            face_detected=face_has_bbox or pose is not None
            face_lm=pose is not None; pose_lm=pose is not None
            if pose is None:
                reason="face bbox present but landmarks missing" if face_has_bbox else "face landmarks missing"
                elapsed=time.time()-self.calib_start_time if self.calib_start_time else 0
                if elapsed>self.calib_fallback and len(self.calib_buffer)>=self.calib_min_valid:
                    self._finalize(f"fallback 3s {elapsed:.1f}s")
                elif elapsed>self.calib_timeout and len(self.calib_buffer)>=1:
                    self._finalize(f"timeout 5s {elapsed:.1f}s")
                elif elapsed>self.calib_timeout*2:
                    self._finalize("forced 10s")
                return {"vector":(0,0),"direction":"CALIBRATING","yaw":0,"pitch":0,"roll":0,"calibrated":False,"calib_progress":len(self.calib_buffer),"calib_total":self.calib_frames,"command":"STOP","rejection_reason":reason,"face_detected":face_detected,"face_landmarks":face_lm,"pose_landmarks":pose_lm,"frame_valid":False}
            yaw,pitch,roll=pose
            valid,reason=self._valid(yaw,pitch,roll)
            if valid:
                self.calib_buffer.append((yaw,pitch,roll))
                if len(self.calib_buffer)>=self.calib_frames: self._finalize()
                else:
                    elapsed=time.time()-self.calib_start_time
                    if elapsed>self.calib_fallback and len(self.calib_buffer)>=self.calib_min_valid: self._finalize(f"fallback 3s {elapsed:.1f}s")
                return {"vector":(0,0),"direction":"CALIBRATING","yaw":yaw,"pitch":pitch,"roll":roll,"calibrated":self.calibrated,"calib_progress":len(self.calib_buffer),"calib_total":self.calib_frames,"command":"STOP","face_detected":True,"face_landmarks":True,"pose_landmarks":True,"frame_valid":True} if not self.calibrated else {"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"roll":0,"calibrated":True,"command":"STOP","face_detected":True}
            else:
                elapsed=time.time()-self.calib_start_time
                if elapsed>self.calib_fallback and len(self.calib_buffer)>=self.calib_min_valid:
                    self._finalize(f"fallback {elapsed:.1f}s")
                    return {"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"roll":0,"calibrated":True,"command":"STOP","face_detected":True}
                return {"vector":(0,0),"direction":"CALIBRATING","yaw":yaw,"pitch":pitch,"roll":roll,"calibrated":False,"calib_progress":len(self.calib_buffer),"calib_total":self.calib_frames,"command":"STOP","rejection_reason":reason,"face_detected":True,"face_landmarks":True,"pose_landmarks":True,"frame_valid":False}
        # --- calibrated ---
        if pose is None:
            self.missing_frames+=1
            if self.missing_frames<=self.missing_tolerance:
                return {"vector":self.last_vec,"direction":self.stable_command,"yaw":0,"pitch":0,"roll":0,"calibrated":True,"command":self.stable_command,"raw":self.last_dir,"face_detected":False}
            cmd=self._confirm("STOP")
            return {"vector":(0,0),"direction":cmd,"yaw":0,"pitch":0,"roll":0,"calibrated":True,"command":cmd,"raw":"NO FACE","face_detected":False}
        self.missing_frames=0
        yaw,pitch,roll=pose
        rel_yaw=yaw-self.calib_yaw; rel_pitch=pitch-self.calib_pitch
        if HEAD_DIRECTION_INVERT_X: rel_yaw=-rel_yaw
        if HEAD_DIRECTION_INVERT_Y: rel_pitch=-rel_pitch
        if self.filtered_yaw is None: self.filtered_yaw=rel_yaw; self.filtered_pitch=rel_pitch
        else:
            a=self.smooth_alpha
            self.filtered_yaw=a*rel_yaw+(1-a)*self.filtered_yaw
            self.filtered_pitch=a*rel_pitch+(1-a)*self.filtered_pitch
        self.history.append((self.filtered_yaw,self.filtered_pitch))
        if len(self.history)>self.hist_len: self.history.pop(0)
        avg_y=float(np.mean([p[0] for p in self.history])); avg_p=float(np.mean([p[1] for p in self.history]))
        self.last_vec=(float(np.clip(avg_y/30,-1,1)),float(np.clip(avg_p/30,-1,1)))
        cand=self._hyst(avg_y,avg_p); self.last_dir=cand
        if cand=="STOP" and self.stable_command=="STOP": 
            a=FACE_CALIBRATE_SMOOTHING
            self.calib_yaw=(1-a)*self.calib_yaw+a*yaw; self.calib_pitch=(1-a)*self.calib_pitch+a*pitch
        cmd=self._confirm(cand)
        stop_reason="" if cmd!="STOP" else (f"candidate {cand} {self.pending_frames}/{self.confirm_frames}" if cand!="STOP" else "threshold not reached")
        if DEBUG_DIRECTION:
            print(f"[DEBUG] yaw={avg_y:.1f} pitch={avg_p:.1f} body_angle={yaw:.1f} face_direction={cand} pose_direction={cand} candidate_direction={cand} stable_frames={self.pending_frames} final_direction={cmd} reason=\"{stop_reason}\"")
        return {"vector":self.last_vec,"direction":cmd,"yaw":avg_y,"pitch":avg_p,"roll":roll,"calibrated":True,"command":cmd,"raw":cand,"rel_yaw":rel_yaw,"rel_pitch":rel_pitch,"face_detected":True,"candidate_count":self.pending_frames,"confirm_threshold":self.confirm_frames,"candidate_direction":cand}

    def close(self): 
        try: self.face_mesh.close()
        except: pass

# ---------- BODY ----------
class BodyDirectionTracker:
    def __init__(self):
        if not MP_AVAILABLE: raise RuntimeError("MediaPipe required")
        self.pose=mp.solutions.pose.Pose(static_image_mode=False,model_complexity=0,enable_segmentation=False,min_detection_confidence=float(BODY_POSE_CONF_THRESH),min_tracking_confidence=float(BODY_POSE_CONF_THRESH))
        self.history=[]; self.hist_len=int(BODY_DIRECTION_HISTORY)
        self.yaw_enter=float(BODY_YAW_ENTER_THRESHOLD); self.yaw_exit=float(BODY_YAW_EXIT_THRESHOLD)
        self.pitch_enter=float(BODY_PITCH_ENTER_THRESHOLD); self.pitch_exit=float(BODY_PITCH_EXIT_THRESHOLD)
        self.offset_enter=float(BODY_SHOULDER_OFFSET_ENTER); self.offset_exit=float(BODY_SHOULDER_OFFSET_EXIT)
        self.smooth_alpha=float(BODY_SMOOTH_ALPHA)
        self.filtered_yaw=None; self.filtered_pitch=None
        self.last_vec=(0.0,0.0); self.last_dir="STOP"; self.stable_command="STOP"; self.pending_command="STOP"; self.pending_frames=0; self.missing_frames=0
        self.confirm_frames=int(BODY_COMMAND_CONFIRM_FRAMES); self.missing_tolerance=int(BODY_MISSING_TOLERANCE)
        self.calib_yaw=None; self.calib_pitch=None; self.calib_offset=None; self.calib_buffer=[]; self.calibrated=False; self.calib_frames=int(BODY_CALIB_FRAMES); self.calib_timeout=float(BODY_CALIB_TIMEOUT_SEC); self.calib_start_time=None; self._perf_idx=0; self._cached_pose=None
        self.L_SHO=11; self.R_SHO=12; self.L_HIP=23; self.R_HIP=24; self.NOSE=0
    def reset_calibration(self):
        self.calib_yaw=self.calib_pitch=self.calib_offset=None; self.calib_buffer=[]; self.calibrated=False
        self.history.clear(); self.filtered_yaw=self.filtered_pitch=None
        self.last_vec=(0.0,0.0); self.last_dir="STOP"; self.stable_command="STOP"; self.pending_command="STOP"; self.pending_frames=0; self.missing_frames=0; self.calib_start_time=None; self._cached_pose=None
        print("[BodyDirection] RESET")
    def _measure_body_pose(self,frame_bgr):
        h,w=frame_bgr.shape[:2]
        if w>320 or h>240: small=cv2.resize(frame_bgr,(256,192),interpolation=cv2.INTER_LINEAR)
        else: small=frame_bgr
        rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
        res=self.pose.process(rgb)
        if not res.pose_landmarks: return None
        lm=res.pose_landmarks.landmark
        if lm[self.L_SHO].visibility<0.5 or lm[self.R_SHO].visibility<0.5: return None
        hips_visible=lm[self.L_HIP].visibility>0.3 and lm[self.R_HIP].visibility>0.3
        l_sho=lm[self.L_SHO]; r_sho=lm[self.R_SHO]; l_hip=lm[self.L_HIP]; r_hip=lm[self.R_HIP]
        sho_cx=(l_sho.x+r_sho.x)/2
        offset=sho_cx-((l_hip.x+r_hip.x)/2) if hips_visible else 0.0
        shoulder_width=abs(l_sho.x-r_sho.x)
        lz=l_sho.z; rz=r_sho.z
        try:
            if res.pose_world_landmarks:
                wlm=res.pose_world_landmarks.landmark
                lz=wlm[self.L_SHO].z; rz=wlm[self.R_SHO].z
                shoulder_width=abs(wlm[self.R_SHO].x-wlm[self.L_SHO].x)+1e-6
        except: pass
        width_safe=max(float(shoulder_width),0.01)
        norm_turn=(rz-lz)/width_safe
        if config_mod.BODY_DIRECTION_INVERT: norm_turn=-norm_turn; offset=-offset
        body_yaw=float(norm_turn*60.0)
        if abs(offset)>0.01: body_yaw=0.7*body_yaw+0.3*offset*600.0
        if hips_visible:
            body_pitch=float(((l_sho.y+r_sho.y)/2 - (l_hip.y+r_hip.y)/2)*-100)
            vis=float(np.mean([lm[i].visibility for i in [self.L_SHO,self.R_SHO,self.L_HIP,self.R_HIP]]))
        else:
            body_pitch=0.0
            vis=float(np.mean([lm[i].visibility for i in [self.L_SHO,self.R_SHO]]))
        self._last_shoulder_l=(l_sho.x,l_sho.y,lz); self._last_shoulder_r=(r_sho.x,r_sho.y,rz)
        self._last_normalized_turn=norm_turn; self._last_shoulder_width=width_safe
        return (body_yaw,body_pitch,offset,vis,vis)
    def _valid(self,yaw,pitch,offset,vis):
        if vis<0.6: return False,f"visibility low {vis:.2f}"
        if abs(yaw)>25: return False,f"yaw {yaw:.1f}>25"
        if abs(pitch)>25: return False,f"pitch {pitch:.1f}>25"
        return True,""
    def _finalize(self,reason=""):
        if len(self.calib_buffer)>=1:
            self.calib_yaw=float(np.median([p[0] for p in self.calib_buffer]))
            self.calib_pitch=float(np.median([p[1] for p in self.calib_buffer]))
            self.calib_offset=float(np.median([p[2] for p in self.calib_buffer]))
            print(f"[BodyDirection] CALIBRATED yaw={self.calib_yaw:.1f} offset={self.calib_offset:.3f} ({len(self.calib_buffer)}/{self.calib_frames}) {reason}")
        else:
            self.calib_yaw=self.calib_pitch=self.calib_offset=0.0
            print(f"[BodyDirection] Fallback 0 {reason}")
        self.calibrated=True; self.history.clear(); self.filtered_yaw=self.filtered_pitch=None
    def _raw(self,ay,ap,off):
        if ay<=-self.yaw_enter or off<=-self.offset_enter: return "LEFT"
        if ay>=self.yaw_enter or off>=self.offset_enter: return "RIGHT"
        if ap<=-self.pitch_enter: return "FORWARD"
        if ap>=self.pitch_enter: return "BACKWARD"
        return "STOP"
    def _hyst(self,ay,ap,off):
        sc=self.stable_command
        if sc=="LEFT" and (ay<=-self.yaw_exit or off<=-self.offset_exit): return "LEFT"
        if sc=="RIGHT" and (ay>=self.yaw_exit or off>=self.offset_exit): return "RIGHT"
        if sc=="FORWARD" and ap<=-self.pitch_exit: return "FORWARD"
        if sc=="BACKWARD" and ap>=self.pitch_exit: return "BACKWARD"
        return self._raw(ay,ap,off)
    def _confirm(self,cand):
        if cand==self.pending_command: self.pending_frames+=1
        else: self.pending_command=cand; self.pending_frames=1
        if self.pending_frames>=self.confirm_frames: self.stable_command=cand
        return self.stable_command
    def detect(self,frame_bgr,face_bbox=None):
        if not ENABLE_BODY_POSE:
            return {"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"calibrated":False,"command":"STOP"}
        if self.calib_start_time is None: self.calib_start_time=time.time()
        self._perf_idx+=1
        use_cached=False
        if self.calibrated and PERFORMANCE_MODE and self._cached_pose is not None and self.missing_frames==0:
            n=CFG.get("face_mesh_every_n",2)
            if n>1 and (self._perf_idx % n)!=0: use_cached=True
        pose=self._cached_pose if use_cached else self._measure_body_pose(frame_bgr)
        if not use_cached:
            if pose is not None: self._cached_pose=pose
        # calibration
        if not self.calibrated:
            if pose is None:
                elapsed=time.time()-self.calib_start_time
                if elapsed>3.0 and len(self.calib_buffer)>=5:
                    self._finalize(f"fallback 3s {elapsed:.1f}s")
                    return {"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"calibrated":True,"command":"STOP","face_detected":face_bbox is not None}
                if elapsed>self.calib_timeout:
                    if len(self.calib_buffer)>=1: self._finalize(f"timeout {elapsed:.1f}s")
                    else: print(f"[BodyCalibration] Failed pose landmarks {elapsed:.1f}s")
                    return {"vector":(0,0),"direction":"CALIBRATING","yaw":0,"pitch":0,"calibrated":False,"command":"STOP","calib_progress":len(self.calib_buffer),"calib_total":self.calib_frames,"face_detected":face_bbox is not None,"rejection_reason":"pose landmarks unavailable"}
                return {"vector":(0,0),"direction":"CALIBRATING","yaw":0,"pitch":0,"calibrated":False,"command":"STOP","calib_progress":len(self.calib_buffer),"calib_total":self.calib_frames,"face_detected":face_bbox is not None}
            yaw,pitch,off,vis,conf=pose
            valid,reason=self._valid(yaw,pitch,off,vis)
            if valid:
                self.calib_buffer.append((yaw,pitch,off))
                if len(self.calib_buffer)>=self.calib_frames:
                    self._finalize()
                    return {"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"calibrated":True,"command":"STOP","face_detected":True}
                elapsed=time.time()-self.calib_start_time
                if elapsed>3.0 and len(self.calib_buffer)>=8:
                    self._finalize(f"fallback 3s {elapsed:.1f}s")
                    return {"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"calibrated":True,"command":"STOP","face_detected":True}
                return {"vector":(0,0),"direction":"CALIBRATING","yaw":yaw,"pitch":pitch,"calibrated":False,"command":"STOP","calib_progress":len(self.calib_buffer),"calib_total":self.calib_frames,"face_detected":True}
            else:
                return {"vector":(0,0),"direction":"CALIBRATING","yaw":yaw,"pitch":pitch,"calibrated":False,"command":"STOP","calib_progress":len(self.calib_buffer),"calib_total":self.calib_frames,"face_detected":True,"rejection_reason":reason}
        # calibrated
        if pose is None:
            self.missing_frames+=1
            if self.missing_frames<=self.missing_tolerance:
                return {"vector":self.last_vec,"direction":self.stable_command,"yaw":0,"pitch":0,"calibrated":True,"command":self.stable_command,"raw":self.last_dir}
            cmd=self._confirm("STOP")
            return {"vector":(0,0),"direction":cmd,"yaw":0,"pitch":0,"calibrated":True,"command":cmd,"raw":"NO BODY"}
        self.missing_frames=0
        yaw,pitch,off,vis,conf=pose
        rel_yaw=yaw-(self.calib_yaw or 0); rel_pitch=pitch-(self.calib_pitch or 0); rel_off=off-(self.calib_offset or 0)
        if self.filtered_yaw is None: self.filtered_yaw=rel_yaw; self.filtered_pitch=rel_pitch
        else:
            a=self.smooth_alpha
            self.filtered_yaw=a*rel_yaw+(1-a)*self.filtered_yaw
            self.filtered_pitch=a*rel_pitch+(1-a)*self.filtered_pitch
        self.history.append((self.filtered_yaw,self.filtered_pitch))
        if len(self.history)>self.hist_len: self.history.pop(0)
        avg_y=float(np.mean([p[0] for p in self.history])); avg_p=float(np.mean([p[1] for p in self.history]))
        self.last_vec=(float(np.clip(avg_y/30,-1,1)),float(np.clip(avg_p/30,-1,1)))
        cand=self._hyst(avg_y,avg_p,rel_off); self.last_dir=cand
        cmd=self._confirm(cand)
        if DEBUG_DIRECTION:
            reason="" if cmd!="STOP" else ("candidate not confirmed" if cand!="STOP" else "threshold not reached")
            if cmd=="STOP" and cand!="STOP": reason=f"candidate {cand} {self.pending_frames}/{self.confirm_frames}"
            print(f"[DEBUG] yaw={avg_y:.1f} pitch={avg_p:.1f} offset={rel_off:.3f} body_angle={yaw:.1f} candidate={cand} stable={self.pending_frames} final={cmd} reason=\"{reason}\"")
        l_sho=getattr(self,'_last_shoulder_l',(0,0,0)); r_sho=getattr(self,'_last_shoulder_r',(0,0,0))
        norm=getattr(self,'_last_normalized_turn',0)
        return {"vector":self.last_vec,"direction":cmd,"yaw":avg_y,"pitch":avg_p,"offset":rel_off,"body_angle":yaw,"body_turn_value":norm,"left_shoulder":l_sho,"right_shoulder":r_sho,"shoulder_depth_diff":float(r_sho[2]-l_sho[2]),"calibrated":True,"command":cmd,"raw":cand,"candidate_direction":cand,"candidate_count":self.pending_frames,"confirm_threshold":self.confirm_frames}

    def close(self):
        try: self.pose.close()
        except: pass

HeadPoseTracker=FaceDirectionTracker

class MediaPipeTracker:
    def __init__(self,complexity=0):
        if not MP_AVAILABLE: raise RuntimeError("mediapipe")
        self.mp_hands=mp.solutions.hands
        self.hands=self.mp_hands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=complexity,min_detection_confidence=MEDIAPIPE_DETECTION_CONF,min_tracking_confidence=MEDIAPIPE_TRACKING_CONF)
    def detect(self,rgb_small):
        res=self.hands.process(rgb_small)
        if not res.multi_hand_landmarks: return None
        lm=res.multi_hand_landmarks[0].landmark[8]
        h,w=rgb_small.shape[:2]
        return (int(lm.x*w),int(lm.y*h))
    def close(self): self.hands.close()

class ColorTracker:
    def __init__(self,lower=HSV_LOWER,upper=HSV_UPPER):
        self.lower=np.array(lower,dtype=np.uint8); self.upper=np.array(upper,dtype=np.uint8)
        self.kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    def detect(self,bgr_small):
        hsv=cv2.cvtColor(bgr_small,cv2.COLOR_BGR2HSV)
        mask=cv2.inRange(hsv,self.lower,self.upper)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,self.kernel)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        c=max(contours,key=cv2.contourArea)
        if cv2.contourArea(c)<MIN_CONTOUR_AREA*(bgr_small.shape[0]*bgr_small.shape[1])/(320*240): return None
        M=cv2.moments(c)
        if M["m00"]==0: return None
        return (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))

class HybridTracker:
    def __init__(self,cfg=CFG,backend="auto",smoother_alpha=0.6,smoother_beta=0.15,face_detector="auto"):
        self.cfg=cfg; self.backend_name=backend; self.face_detector_name=face_detector
        from config import FACE_SMOOTHER_ALPHA
        self.smoother=Smoother(alpha=smoother_alpha,beta=smoother_beta)
        self.face_smoother=Smoother(alpha=FACE_SMOOTHER_ALPHA,beta=0.30)
        self.det_w=cfg["detect_width"]; self.det_h=cfg["detect_height"]
        self.cam_w=cfg["camera_width"]; self.cam_h=cfg["camera_height"]
        self.scale_x=self.cam_w/self.det_w; self.scale_y=self.cam_h/self.det_h
        self.tracker=None; self.face_tracker=None; self.head_pose_tracker=None; self.body_pose_tracker=None
        if ENABLE_HEAD_POSE:
            try:
                self.head_pose_tracker=FaceDirectionTracker()
                print("[FaceDirection] ON")
            except Exception as e:
                print(f"[FaceDirection] Disabled {e}")
        if ENABLE_BODY_POSE:
            try:
                self.body_pose_tracker=BodyDirectionTracker()
                print("[BodyDirection] ON - primary")
            except Exception as e:
                print(f"[BodyDirection] Disabled {e}")
        self._init_backend()
        self.frame_count=0; self.last_pt=None; self.last_face=None; self.last_detect_time=0; self.last_face_center=None
        self.last_head_pose={"vector":(0,0),"direction":"NO FACE","yaw":0,"pitch":0,"roll":0,"command":"STOP","calibrated":False} if self.head_pose_tracker else {"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"roll":0,"command":"STOP","calibrated":True}
        self.last_body_pose={"vector":(0,0),"direction":"NO BODY","yaw":0,"pitch":0,"calibrated":False,"command":"STOP"} if self.body_pose_tracker else {"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"calibrated":True,"command":"STOP"}
        if self.head_pose_tracker is None and ENABLE_HEAD_POSE:
            self.last_head_pose={"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"roll":0,"command":"STOP","calibrated":True}
        if self.body_pose_tracker is None and ENABLE_BODY_POSE:
            self.last_body_pose={"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"calibrated":True,"command":"STOP"}
    def _init_backend(self):
        backend=self.backend_name
        if backend=="auto": backend="face"
        if backend=="face":
            try:
                fd=self.face_detector_name
                if fd=="mediapipe" and MP_AVAILABLE:
                    from tracker import MediaPipeFaceTracker
                    self.face_tracker=MediaPipeFaceTracker(); self.face_detector_name="mediapipe"
                elif fd=="haar" or fd=="auto":
                    self.face_tracker=FaceTrackerHaar(); self.face_detector_name="haar"
                else:
                    self.face_tracker=FaceTrackerHaar(); self.face_detector_name="haar"
                print(f"[Tracker] Face {self.face_detector_name}")
            except Exception as e:
                print(f"[Tracker] Face fail {e}"); self.face_tracker=FaceTrackerHaar(); self.face_detector_name="haar"
            if MP_AVAILABLE:
                try: self.tracker=MediaPipeTracker(complexity=self.cfg.get("mediapipe_complexity",0))
                except: self.tracker=ColorTracker()
            else: self.tracker=ColorTracker()
            self.backend_name="face"
        elif backend=="mediapipe":
            if not MP_AVAILABLE:
                self.tracker=ColorTracker(); self.backend_name="color"
            else:
                try: self.tracker=MediaPipeTracker(complexity=self.cfg.get("mediapipe_complexity",0)); self.backend_name="mediapipe"
                except: self.tracker=ColorTracker(); self.backend_name="color"
        elif backend=="color":
            self.tracker=ColorTracker(); self.backend_name="color"; print("[Tracker] Color")
        else: raise ValueError(backend)
    def _detect_face(self,frame_bgr):
        if frame_bgr.shape[1]==self.det_w and frame_bgr.shape[0]==self.det_h: small=frame_bgr
        else: small=cv2.resize(frame_bgr,(self.det_w,self.det_h),interpolation=cv2.INTER_LINEAR)
        if self.face_detector_name=="mediapipe":
            rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
            return self.face_tracker.detect(rgb)
        else:
            gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
            return self.face_tracker.detect(gray)
    def update(self,frame_bgr):
        self.frame_count+=1; n=self.cfg.get("detect_every_n_frames",1)
        if (self.frame_count % n)!=0 and (self.last_face is not None or self.last_pt is not None):
            if (self.head_pose_tracker and not self.head_pose_tracker.calibrated) or (self.body_pose_tracker and not self.body_pose_tracker.calibrated):
                try:
                    if self.head_pose_tracker and not self.head_pose_tracker.calibrated:
                        self.last_head_pose=self.head_pose_tracker.detect(frame_bgr,face_bbox=self.last_face)
                    if self.body_pose_tracker and not self.body_pose_tracker.calibrated:
                        self.last_body_pose=self.body_pose_tracker.detect(frame_bgr,face_bbox=self.last_face)
                except: pass
                calib=self.last_head_pose if self.head_pose_tracker and not self.head_pose_tracker.calibrated else self.last_body_pose
                return {"face":self.last_face,"face_center":self.face_smoother.update(self.last_face_center) if self.last_face_center else None,"hand":self.smoother.update(self.last_pt) if self.last_pt else None,"low_light":False,"head_pose":calib,"body_pose":self.last_body_pose,"primary_vec":calib.get("vector",(0,0)),"primary_dir":calib.get("direction","CALIBRATING"),"command":calib.get("command","STOP")}
            return {"face":self.last_face,"face_center":self.face_smoother.update(self.last_face_center) if self.last_face_center else None,"hand":self.smoother.update(self.last_pt) if self.last_pt else None,"low_light":False,"head_pose":self.last_head_pose,"body_pose":self.last_body_pose,"primary_vec":self.last_head_pose.get("vector",(0,0)),"primary_dir":self.last_head_pose.get("direction","STOP"),"command":self.last_head_pose.get("command","STOP")}
        t0=time.perf_counter(); face_small=None; hand_small=None
        if self.backend_name=="face":
            face_small=self._detect_face(frame_bgr)
            if face_small is None and self.tracker is not None:
                if frame_bgr.shape[1]==self.det_w and frame_bgr.shape[0]==self.det_h: small=frame_bgr
                else: small=cv2.resize(frame_bgr,(self.det_w,self.det_h),interpolation=cv2.INTER_LINEAR)
                if isinstance(self.tracker,MediaPipeTracker):
                    rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB); hand_small=self.tracker.detect(rgb)
                else: hand_small=self.tracker.detect(small)
        elif self.backend_name=="mediapipe":
            if frame_bgr.shape[1]==self.det_w and frame_bgr.shape[0]==self.det_h: small=frame_bgr
            else: small=cv2.resize(frame_bgr,(self.det_w,self.det_h),interpolation=cv2.INTER_LINEAR)
            rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB); hand_small=self.tracker.detect(rgb)
        else:
            if frame_bgr.shape[1]==self.det_w and frame_bgr.shape[0]==self.det_h: small=frame_bgr
            else: small=cv2.resize(frame_bgr,(self.det_w,self.det_h),interpolation=cv2.INTER_LINEAR)
            hand_small=self.tracker.detect(small)
        self.last_detect_time=(time.perf_counter()-t0)*1000
        face_cam=None; face_center=None
        if face_small is not None:
            x,y,w,h=face_small
            x_cam=int(x*self.scale_x); y_cam=int(y*self.scale_y); w_cam=int(w*self.scale_x); h_cam=int(h*self.scale_y)
            x_cam=max(0,min(self.cam_w-1,x_cam)); y_cam=max(0,min(self.cam_h-1,y_cam))
            w_cam=max(10,min(self.cam_w-x_cam,w_cam)); h_cam=max(10,min(self.cam_h-y_cam,h_cam))
            face_cam=(x_cam,y_cam,w_cam,h_cam); self.last_face=face_cam; cx=x_cam+w_cam//2; cy=y_cam+h_cam//2; self.last_face_center=(cx,cy); face_center=self.face_smoother.update(self.last_face_center)
        head_pose=self.last_head_pose
        if self.head_pose_tracker is not None:
            head_pose=self.head_pose_tracker.detect(frame_bgr,face_bbox=face_cam); self.last_head_pose=head_pose
        elif ENABLE_HEAD_POSE:
            head_pose={"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"roll":0,"command":"STOP","calibrated":True,"face_detected":face_cam is not None}
        body_pose=self.last_body_pose
        if self.body_pose_tracker is not None:
            try: body_pose=self.body_pose_tracker.detect(frame_bgr,face_bbox=face_cam); self.last_body_pose=body_pose
            except: body_pose=self.last_body_pose
        elif ENABLE_BODY_POSE:
            body_pose={"vector":(0,0),"direction":"STOP","yaw":0,"pitch":0,"calibrated":True,"command":"STOP"}
        # Fusion: BODY is primary as per spec - ALWAYS use body when body tracker is calibrated
        # This guarantees body LEFT/RIGHT always shows, not overwritten by face STOP
        # Fallback to face only if body tracker is disabled or not calibrated
        body_valid=self.body_pose_tracker is not None and body_pose.get("calibrated") and body_pose.get("direction")!="CALIBRATING"
        head_valid=head_pose.get("calibrated") and head_pose.get("direction")!="CALIBRATING"
        use_body = bool(body_valid)
        if use_body:
            primary_vec=body_pose["vector"]; primary_dir=body_pose["direction"]; command=body_pose["command"]
        else:
            if head_valid:
                primary_vec=head_pose["vector"]; primary_dir=head_pose["direction"]; command=head_pose.get("command","STOP")
            else:
                primary_vec=(0,0); primary_dir="STOP"; command="STOP"
        hand_cam=None
        if hand_small is not None:
            x_cam=int(hand_small[0]*self.scale_x); y_cam=int(hand_small[1]*self.scale_y)
            self.last_pt=(x_cam,y_cam); hand_cam=self.smoother.update(self.last_pt)
        if self.backend_name=="face":
            overall=head_pose.get("calibrated",True) and body_pose.get("calibrated",True)
            if not hasattr(self,'_was_calibrated'): self._was_calibrated=False
            if overall and not self._was_calibrated:
                nyaw=self.head_pose_tracker.calib_yaw if self.head_pose_tracker and self.head_pose_tracker.calib_yaw is not None else 0
                nbody=self.body_pose_tracker.calib_offset if self.body_pose_tracker and self.body_pose_tracker.calib_offset is not None else 0
                print(f"CALIBRATION COMPLETE NEUTRAL YAW: {nyaw:.1f} NEUTRAL BODY TURN: {nbody:.3f}")
                self._was_calibrated=True
            elif not overall: self._was_calibrated=False
            ui_pose=head_pose if not head_pose.get("calibrated",True) else (body_pose if not body_pose.get("calibrated",True) else head_pose)
            return {"face":face_cam if face_cam is not None else self.last_face if self.frame_count%6!=0 else None,"face_center":face_center if face_center is not None else (self.face_smoother.update(self.last_face_center) if self.last_face_center and face_cam is None else None),"hand":hand_cam,"head_pose":ui_pose if not overall else head_pose,"body_pose":body_pose,"head_pose_raw":head_pose,"primary_vec":primary_vec,"primary_dir":primary_dir,"command":command,"calibrated":overall,"low_light":False}
        if hand_cam is None: return None
        return hand_cam
    def update_legacy(self,frame_bgr):
        res=self.update(frame_bgr)
        if isinstance(res,dict): return res.get("hand") or res.get("face_center")
        return res
    def get_last_detect_latency_ms(self): return self.last_detect_time
    def reset_calibration(self):
        if self.head_pose_tracker: self.head_pose_tracker.reset_calibration()
        if self.body_pose_tracker: self.body_pose_tracker.reset_calibration()
    def close(self):
        if hasattr(self.tracker,'close'):
            try: self.tracker.close()
            except: pass
        if self.face_tracker and hasattr(self.face_tracker,'close'):
            try: self.face_tracker.close()
            except: pass
        if hasattr(self,'head_pose_tracker') and self.head_pose_tracker and hasattr(self.head_pose_tracker,'close'):
            try: self.head_pose_tracker.close()
            except: pass
        if hasattr(self,'body_pose_tracker') and self.body_pose_tracker and hasattr(self.body_pose_tracker,'close'):
            try: self.body_pose_tracker.close()
            except: pass
