# media_controller.py
# Goal:
#   - Live gesture control for media keys.
#   - Uses HSV∨YCrCb masks and optional MediaPipe to focus ROI.
#   - Tracks centroid trail and radians-based rotation (cw/ccw).
#   - Consumes features: theta_rad, dtheta_rad, theta_accum_rad, rot_dir.

import time, math
from pathlib import Path
from collections import deque
import numpy as np
import cv2
import joblib
import pyautogui

# Try MediaPipe for dynamic hand ROI (auto if available)
MP_OK = True
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
except Exception:
    MP_OK = False

# Model files (scaler, classifier, optional label map, feature order)
SCALER_PKL = Path("gesture_scaler.pkl")
MODEL_PKL  = Path("gesture_model.pkl")
LABEL_MAP  = Path("label_map.npy")
FEAT_NAMES = Path("feature_names.npy")  # rotation cols must exist or will be appended

# Main config (camera, masks, ROI, stability, rotation)
CFG = dict(
    CAM_INDEX=0, CAP_FPS=30, CAP_WIDTH=960, CAP_HEIGHT=540,

    PREFER="and",            # "and" | "hsv" | "ycc"

    # static ROI when MP is not used
    ROI_SIDE="right",        # "none" | "right" | "left"
    ROI_FRAC=0.45,
    FACE_TOP_GUARD=0.32,
    MIN_AREA_PX=1600,
    MAX_AREA_FRAC=0.55,

    # dynamic ROI via MediaPipe (auto mode)
    BACKEND_AUTO=True,
    ROI_PAD_FRAC=0.25,
    ROI_MIN_DET_CONF=0.45,
    ROI_MIN_TRK_CONF=0.35,

    SHOW_HUD=1, PRINT_HZ=4.0,

    # label stability / debouncing
    HIST_LEN=5,
    MAJORITY=3,
    HOLD_MS=220,
    GLOBAL_COOLDOWN_S=0.30,

    # continuous volume repeat
    VOL_INTERVAL_S=0.35,

    # confidence switching
    MIN_CONF_SWITCH=0.55,
    DROP_DECAY=0.6,

    # trails
    TRAIL_LEN=24,
    TRAIL_DECAY=0.92,
    TRAIL_THICK=3,
    TRAIL_MIN_NRM=0.002,

    # radians rotation around EMA center
    ROT_ENABLE=True,
    ROT_CENTER_EMA=0.15,
    ROT_MIN_RADIUS_FRAC=0.04,
    ROT_TRIGGER_RAD=2.2,
    ROT_DECAY=0.98,
    ROT_RESET_AFTER_FIRE=True,
)

# Map canonical labels to media actions
CANON = {
    "open_palm":"play","open":"play","five":"play","palm":"play","play":"play",
    "pause":"pause","closed_fist":"pause","closedfist":"pause","fist":"pause","stop":"pause",
    "thumb_up":"vol_up","thumbs_up":"vol_up","thumbup":"vol_up","up":"vol_up",
    "volume_up":"vol_up","vol_up":"vol_up","volup":"vol_up",
    "thumb_down":"vol_down","thumbs_down":"vol_down","thumbdown":"vol_down","down":"vol_down",
    "volume_down":"vol_down","vol_down":"vol_down","voldown":"vol_down",
    "cw":"next","clockwise":"next","rotate_cw":"next","spin_right":"next",
    "next":"next","forward":"next","nexttrack":"next","seek_right":"next",
    "ccw":"prev","counter_clockwise":"prev","counterclockwise":"prev","rotate_ccw":"prev","spin_left":"prev",
    "prev":"prev","previous":"prev","prevtrack":"prev","rewind":"prev","seek_left":"prev",
}

# CSV19 base + rotation columns
CSV19_BASE = [
    "time_s","w","h","cx","cy","cx_n","cy_n",
    "area","area_norm","solidity","aspect_ratio","circularity",
    "bbox_x","bbox_y","bbox_w","bbox_h","extent",
    "finger_count","prior_nboxes"
]
ROT_COLS = ["theta_rad","dtheta_rad","theta_accum_rad","rot_dir"]

# Mask thresholds and kernels
HSV1_LO = np.array([  0,  25,  45], np.uint8)
HSV1_HI = np.array([ 25, 255, 255], np.uint8)
HSV2_LO = np.array([170,  20,  45], np.uint8)
HSV2_HI = np.array([179, 255, 255], np.uint8)
YCC_LO  = np.array([ 10, 133,  77], np.uint8)
YCC_HI  = np.array([255, 173, 127], np.uint8)
KERNEL       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

def put(img, x, y, s, col=(40,255,40)):
    cv2.putText(img, s, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

def masked_blend(bgr, mask_u8, color=(0,255,0), alpha=0.35):
    overlay = np.full_like(bgr, color, np.uint8)
    blend = cv2.addWeighted(bgr, 1.0, overlay, alpha, 0.0)
    out = bgr.copy(); m = mask_u8.astype(bool); out[m] = blend[m]; return out

def gray_world_wb(bgr):
    b,g,r = cv2.split(bgr.astype(np.float32))
    mb,mg,mr = b.mean()+1e-6, g.mean()+1e-6, r.mean()+1e-6
    m = (mb+mg+mr)/3.0
    b*=m/mb; g*=m/mg; r*=m/mr
    return np.clip(cv2.merge([b,g,r]),0,255).astype(np.uint8)

def adaptive_gamma_from_Y(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:,:,0].astype(np.float32)/255.0
    gamma = float(np.clip(1.4 - float(np.clip(Y.mean(),1e-3,0.999)), 0.7, 1.3))
    inv = 1.0/gamma
    lut = np.array([(i/255.0)**inv*255.0 for i in range(256)], np.uint8)
    return cv2.LUT(bgr, lut)

def preprocess_shared(bgr):
    return cv2.GaussianBlur(adaptive_gamma_from_Y(gray_world_wb(bgr)), (5,5), 0)

def postprocess_mask(m):
    m = (m>0).astype(np.uint8)*255
    m = cv2.erode(m, KERNEL, 1)
    m = cv2.dilate(m, KERNEL, 2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL_CLOSE, 1)
    return m

def apply_roi(mask, frac, side):
    if side=="none" or frac<=0: return mask
    h,w = mask.shape[:2]; xcut = int(w*np.clip(frac,0.0,1.0))
    roi = np.zeros_like(mask)
    if side=="right": roi[:, xcut:] = 255
    elif side=="left": roi[:, :w-xcut] = 255
    else: return mask
    return cv2.bitwise_and(mask, roi)

def face_top_guard(mask, frac):
    if frac<=0: return mask
    h,w = mask.shape[:2]; guard = np.zeros_like(mask); guard[int(h*frac):,:] = 255
    return cv2.bitwise_and(mask, guard)

def rect_to_mask(shape, x0, y0, x1, y1):
    h, w = shape[:2]
    x0 = int(np.clip(x0, 0, w-1)); x1 = int(np.clip(x1, 0, w-1))
    y0 = int(np.clip(y0, 0, h-1)); y1 = int(np.clip(y1, 0, h-1))
    m = np.zeros((h,w), np.uint8)
    if x1>x0 and y1>y0: m[y0:y1, x0:x1] = 255
    return m

def build_masks_shared(bgr, roi_frac, roi_side, top_guard, dyn_roi_mask=None):
    blur = preprocess_shared(bgr)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    m_h = postprocess_mask(cv2.bitwise_or(cv2.inRange(hsv, HSV1_LO, HSV1_HI),
                                          cv2.inRange(hsv, HSV2_LO, HSV2_HI)))
    ycc = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    m_y = postprocess_mask(cv2.inRange(ycc, YCC_LO, YCC_HI))
    if dyn_roi_mask is not None:
        m_h = cv2.bitwise_and(m_h, dyn_roi_mask)
        m_y = cv2.bitwise_and(m_y, dyn_roi_mask)
    else:
        m_h = apply_roi(m_h, roi_frac, roi_side)
        m_y = apply_roi(m_y, roi_frac, roi_side)
    if top_guard>0:
        m_h = face_top_guard(m_h, top_guard)
        m_y = face_top_guard(m_y, top_guard)
    return m_h, m_y

def largest_blob(mask, min_area, max_area_abs):
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best=None; best_a=0
    for c in cnts:
        a = cv2.contourArea(c)
        if a>=min_area and a<=max_area_abs and a>best_a:
            best=c; best_a=a
    return best

def mask_score(m, shape):
    h,w = shape[:2]; fa = float(h*w)
    cnt = largest_blob(m, 1200, 0.6*fa)
    if cnt is None: return 0.0
    area = cv2.contourArea(cnt); hull = cv2.convexHull(cnt)
    hull_a = max(cv2.contourArea(hull),1e-6)
    peri = max(cv2.arcLength(cnt, True),1e-6)
    circ = float(np.clip(4.0*np.pi*area/(peri*peri),0.2,1.0))
    return float((area/fa)*(area/hull_a)*circ)

def auto_fuse(hsv_pp, ycc_pp, shape, prefer):
    and_m = cv2.bitwise_and(hsv_pp, ycc_pp)
    or_m  = cv2.bitwise_or(hsv_pp,  ycc_pp)
    s_and = mask_score(and_m, shape)
    s_h   = mask_score(hsv_pp, shape)
    s_y   = mask_score(ycc_pp, shape)
    if prefer=="and" and s_and>=0.0006: return and_m, "AND"
    if s_h>=0.0008 or s_y>=0.0008:
        if s_h>=1.15*s_y: return hsv_pp, "HSV"
        if s_y>=1.15*s_h: return ycc_pp, "YCrCb"
    return or_m, "OR"

def estimate_fingers(cnt):
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx)<3: return 0
    defects = cv2.convexityDefects(cnt, hull_idx)
    if defects is None: return 0
    valleys=0
    for i in range(defects.shape[0]):
        _,_,_,d = defects[i,0]
        if d>800: valleys += 1
    return int(np.clip(valleys+1, 0, 5))

def load_feature_order(path_feat_names):
    # use saved feature order; ensure rotation fields exist
    default = CSV19_BASE + ROT_COLS
    if path_feat_names.exists():
        try:
            names = list(np.load(path_feat_names, allow_pickle=True))
            names_l = [n.lower() for n in names]
            for f in ROT_COLS:
                if f.lower() not in names_l:
                    names.append(f)
            return names
        except Exception:
            return default
    return default

def discover_trained_classes(model, label_map_path):
    if label_map_path.exists():
        try:
            lm = list(np.load(label_map_path, allow_pickle=True))
            return [str(x) for x in lm]
        except Exception:
            pass
    for attr in ("classes_", "classes"):
        if hasattr(model, attr):
            try:
                return [str(x) for x in list(getattr(model, attr))]
            except Exception:
                pass
    return []

def canonicalize_or_passthrough(label_str):
    k = str(label_str).strip().lower().replace(" ", "_")
    return CANON.get(k, k)

def best_with_conf(model, Xs):
    try:
        proba = model.predict_proba(Xs)[0]
        idx = int(np.argmax(proba))
        return idx, float(np.max(proba)), proba
    except Exception:
        pred = model.predict(Xs)[0]
        return pred, None, None

def safe_press(key):
    try:
        pyautogui.press(key); return None
    except Exception as e:
        return f"pyautogui error: {e}"

def mp_get_hand_roi(rgb, pad_frac, min_det=0.5, min_trk=0.5, hands_ctx=None):
    H, W = rgb.shape[:2]
    if hands_ctx is None: return None, None
    res = hands_ctx.process(rgb)
    if not res.multi_hand_landmarks: return None, None
    best_box = None; best_a = 0
    for lm in res.multi_hand_landmarks:
        xs = [int(p.x * W) for p in lm.landmark]
        ys = [int(p.y * H) for p in lm.landmark]
        x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
        bw = max(1, x1 - x0); bh = max(1, y1 - y0); a = bw * bh
        if a > best_a: best_a = a; best_box = (x0, y0, x1, y1)
    if best_box is None: return None, None
    x0,y0,x1,y1 = best_box
    pw = int((x1-x0)*pad_frac); ph = int((y1-y0)*pad_frac)
    x0 -= pw; x1 += pw; y0 -= ph; y1 += ph
    m = rect_to_mask((H,W,3), x0, y0, x1, y1)
    return m, (x0,y0,x1,y1)

def main():
    if not SCALER_PKL.exists() or not MODEL_PKL.exists():
        print("[ERROR] Missing model files."); return

    scaler = joblib.load(SCALER_PKL)
    clf    = joblib.load(MODEL_PKL)
    exp_order = load_feature_order(FEAT_NAMES)

    trained = discover_trained_classes(clf, LABEL_MAP)
    idx_to_name = {i: str(nm) for i,nm in enumerate(trained)}

    try: pyautogui.FAILSAFE = False
    except Exception: pass

    cap = cv2.VideoCapture(CFG["CAM_INDEX"], cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, CFG["CAP_FPS"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG["CAP_WIDTH"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["CAP_HEIGHT"])
    if not cap.isOpened():
        print("ERROR: camera open failed"); return

    # Backend: MP if available and allowed, else pure CV
    use_mp = (MP_OK and CFG["BACKEND_AUTO"])
    hands_ctx = None
    if use_mp:
        hands_ctx = mp_hands.Hands(False, max_num_hands=1, model_complexity=0,
                                   min_detection_confidence=CFG["ROI_MIN_DET_CONF"],
                                   min_tracking_confidence=CFG["ROI_MIN_TRK_CONF"])

    counts = {k:0.0 for k in ["play","pause","vol_up","vol_down","next","prev"]}
    label_hist = deque(maxlen=max(3, CFG["HIST_LEN"]))
    stable_label = None
    first_seen_time = 0.0
    episode_id = 0
    fired_this_episode = set()
    last_action_time = 0.0
    last_vol_time = 0.0
    last_hud_action = "(idle)"

    # trail + rotation state
    trail_pts = deque(maxlen=CFG["TRAIL_LEN"])
    trail_canvas = None
    rot_center = None
    prev_theta = None
    theta_accum_rad = 0.0
    theta_rad = 0.0
    dtheta_rad = 0.0

    last_print = 0.0
    print_interval = 1.0/max(CFG["PRINT_HZ"], 0.1)

    while True:
        ok, frame_raw = cap.read()
        if not ok: break
        raw = cv2.flip(frame_raw, 1)
        H, W = raw.shape[:2]
        diag = (W**2 + H**2) ** 0.5
        if trail_canvas is None:
            trail_canvas = np.zeros((H,W), np.uint8)

        dyn_roi_mask = None
        dyn_box = None
        if use_mp:
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            dyn_roi_mask, dyn_box = mp_get_hand_roi(
                rgb,
                pad_frac=CFG["ROI_PAD_FRAC"],
                min_det=CFG["ROI_MIN_DET_CONF"],
                min_trk=CFG["ROI_MIN_TRK_CONF"],
                hands_ctx=hands_ctx,
            )

        m_h, m_y = build_masks_shared(
            raw, CFG["ROI_FRAC"], CFG["ROI_SIDE"], CFG["FACE_TOP_GUARD"],
            dyn_roi_mask=dyn_roi_mask
        )
        m_f, tag = auto_fuse(m_h, m_y, raw.shape, CFG["PREFER"])

        cnts,_ = cv2.findContours(m_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nboxes = len(cnts)
        cnt = largest_blob(m_f, CFG["MIN_AREA_PX"], CFG["MAX_AREA_FRAC"]*float(H*W))

        vis = masked_blend(raw, m_f if cnt is not None else np.zeros_like(m_f), (0,255,0), 0.35) if CFG["SHOW_HUD"] else raw.copy()
        if CFG["SHOW_HUD"] and dyn_box is not None:
            x0,y0,x1,y1 = dyn_box
            cv2.rectangle(vis, (max(0,x0),max(0,y0)), (min(W-1,x1),min(H-1,y1)), (255,120,30), 2)

        cur = None; conf = None; proba = None

        # centroid + trail + radians rotation
        palm_c = None
        feats = {}
        rot_dir_txt = "-"

        if cnt is not None:
            M = cv2.moments(cnt)
            if abs(M["m00"])>1e-6:
                cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
                palm_c = (cx, cy)
                trail_pts.append(palm_c)

                if rot_center is None:
                    rot_center = (cx, cy)
                    prev_theta = None
                    theta_accum_rad = 0.0

                alpha = float(np.clip(CFG["ROT_CENTER_EMA"],0,1))
                rcx = (1-alpha)*rot_center[0] + alpha*cx
                rcy = (1-alpha)*rot_center[1] + alpha*cy
                rot_center = (rcx, rcy)

                dx = cx - rcx; dy = cy - rcy
                theta_rad = math.atan2(-(dy), dx)  # screen coords (y down)
                radius_ok = (dx*dx + dy*dy)**0.5 >= CFG["ROT_MIN_RADIUS_FRAC"] * diag
                if radius_ok:
                    if prev_theta is not None:
                        dtheta_rad = float((theta_rad - prev_theta + np.pi) % (2*np.pi) - np.pi)
                        theta_accum_rad = theta_accum_rad * CFG["ROT_DECAY"] + dtheta_rad
                        if theta_accum_rad >= CFG["ROT_TRIGGER_RAD"]:
                            rot_dir_txt = "ccw"
                        elif theta_accum_rad <= -CFG["ROT_TRIGGER_RAD"]:
                            rot_dir_txt = "cw"
                        else:
                            rot_dir_txt = "-"
                    prev_theta = theta_rad
                else:
                    theta_accum_rad *= CFG["ROT_DECAY"]
                    rot_dir_txt = "-"

            # geometric features
            area = cv2.contourArea(cnt); area_norm = area/float(H*W)
            hull = cv2.convexHull(cnt); hull_a = max(cv2.contourArea(hull),1e-6)
            solidity = area/hull_a
            x,y,bw,bh = cv2.boundingRect(cnt)
            aspect = bw/max(bh,1)
            peri = max(cv2.arcLength(cnt, True),1e-6)
            circularity = 4.0*np.pi*area/(peri*peri)
            extent = float(area)/max(bw*bh,1)

            feats.update({
                "time_s": float(time.time()),
                "w": float(W), "h": float(H),
                "cx": float(palm_c[0]) if palm_c else 0.0,
                "cy": float(palm_c[1]) if palm_c else 0.0,
                "cx_n": float(palm_c[0])/float(W) if palm_c else 0.0,
                "cy_n": float(palm_c[1])/float(H) if palm_c else 0.0,
                "area": float(area), "area_norm": float(area_norm),
                "solidity": float(solidity), "aspect_ratio": float(aspect),
                "circularity": float(circularity),
                "bbox_x": float(x), "bbox_y": float(y),
                "bbox_w": float(bw), "bbox_h": float(bh),
                "extent": float(extent),
                "finger_count": float(estimate_fingers(cnt)),
                "prior_nboxes": float(nboxes),
                "theta_rad": float(theta_rad),
                "dtheta_rad": float(dtheta_rad),
                "theta_accum_rad": float(theta_accum_rad),
                "rot_dir": float(1.0 if rot_dir_txt=="ccw" else (-1.0 if rot_dir_txt=="cw" else 0.0)),
            })

            # predict current label
            missing = [k for k in exp_order if k not in feats]
            if not missing:
                X = np.array([[feats[k] for k in exp_order]], np.float32)
                try:
                    Xs = scaler.transform(X)
                    pred_idx_or_str, conf, proba = best_with_conf(clf, Xs)
                    if isinstance(pred_idx_or_str, (int, np.integer)) and len(idx_to_name):
                        cur = canonicalize_or_passthrough(idx_to_name[int(pred_idx_or_str)])
                    else:
                        cur = canonicalize_or_passthrough(pred_idx_or_str)
                except Exception as e:
                    cur = None
                    if CFG["SHOW_HUD"]:
                        put(vis, 10, 24, f"model error: {e}", (0,0,255))
            else:
                if CFG["SHOW_HUD"]:
                    put(vis, 10, 24, f"missing: {missing[:5]}...", (0,0,255))

            if CFG["SHOW_HUD"]:
                cv2.drawContours(vis, [cnt], -1, (0,255,255), 2)
                if palm_c is not None:
                    cv2.circle(vis, (int(palm_c[0]), int(palm_c[1])), 5, (0,0,255), -1)
                if rot_center is not None and palm_c is not None:
                    cv2.circle(vis, (int(rot_center[0]), int(rot_center[1])), 5, (255,200,0), -1)
                    cv2.line(vis, (int(rot_center[0]), int(rot_center[1])),
                                   (int(palm_c[0]),     int(palm_c[1])), (255,200,0), 2)

        # update trail buffer and draw last segment
        trail_canvas = (trail_canvas.astype(np.float32)*CFG["TRAIL_DECAY"]).astype(np.uint8)
        if len(trail_pts) >= 2:
            x0,y0 = trail_pts[-2]; x1,y1 = trail_pts[-1]
            if np.hypot(x1-x0,y1-y0)/np.hypot(W,H) >= CFG["TRAIL_MIN_NRM"]:
                cv2.line(trail_canvas,(int(x0),int(y0)),(int(x1),int(y1)),255,CFG["TRAIL_THICK"])
        if CFG["SHOW_HUD"]:
            trail_rgb = np.zeros_like(raw); trail_rgb[:,:,1] = np.clip(trail_canvas,0,255)
            vis = cv2.addWeighted(vis,1.0, trail_rgb,0.55,0)

        # rotation override for next/prev when threshold reached
        if CFG["ROT_ENABLE"] and cnt is not None:
            if theta_accum_rad <= -CFG["ROT_TRIGGER_RAD"]:
                cur = "next"
                if CFG["ROT_RESET_AFTER_FIRE"]: theta_accum_rad = 0.0
            elif theta_accum_rad >=  CFG["ROT_TRIGGER_RAD"]:
                cur = "prev"
                if CFG["ROT_RESET_AFTER_FIRE"]: theta_accum_rad = 0.0

        # soft counts decay
        for k in counts: counts[k] *= CFG["DROP_DECAY"]

        # add vote if we have a prediction
        if cur is not None and cur in counts:
            if stable_label is not None and cur != stable_label and conf is not None and conf >= CFG["MIN_CONF_SWITCH"]:
                for k in counts: counts[k] = 0.0
                label_hist.clear()
                stable_label = None
            counts[cur] += 1.0
            label_hist.append(cur)

        # pick majority label in the recent window
        soft_best = max(counts.items(), key=lambda kv: kv[1])[0] if len(counts) else None
        top = None; best = 0
        for s in set(label_hist):
            c = sum(1 for v in label_hist if v==s)
            if c>best: best=c; top=s

        now = time.time()
        if top is not None and best>=CFG["MAJORITY"]:
            if stable_label!=top:
                stable_label = top
                first_seen_time = now
                episode_id += 1
                fired_this_episode = set()
        else:
            if conf is not None and conf >= CFG["MIN_CONF_SWITCH"] and soft_best is not None:
                stable_label = soft_best
                first_seen_time = now
                episode_id += 1
                fired_this_episode = set()

        # fire discrete actions (play/pause/next/prev) with cooldown
        action_fired = None
        if stable_label is not None:
            held_ms = (now - first_seen_time)*1000.0
            if held_ms >= CFG["HOLD_MS"]:
                can_global = (now - last_action_time) > float(CFG["GLOBAL_COOLDOWN_S"])

                if stable_label == "play" and can_global:
                    if "play" not in fired_this_episode:
                        err = safe_press("play"); action_fired = "play" if err is None else err
                        if err is None:
                            last_action_time = now
                            fired_this_episode.add("play")

                elif stable_label == "pause" and can_global:
                    if "pause" not in fired_this_episode:
                        err = safe_press("pause"); action_fired = "pause" if err is None else err
                        if err is None:
                            last_action_time = now
                            fired_this_episode.add("pause")

                elif stable_label == "next" and can_global:
                    if "next" not in fired_this_episode:
                        err = safe_press("nexttrack"); action_fired = "next" if err is None else err
                        if err is None:
                            last_action_time = now
                            fired_this_episode.add("next")

                elif stable_label == "prev" and can_global:
                    if "prev" not in fired_this_episode:
                        err = safe_press("prevtrack"); action_fired = "prev" if err is None else err
                        if err is None:
                            last_action_time = now
                            fired_this_episode.add("prev")

        # continuous volume (repeat while held)
        if stable_label in ("vol_up","vol_down") and cur == stable_label and cnt is not None:
            if (now - last_vol_time) >= float(CFG["VOL_INTERVAL_S"]):
                if stable_label == "vol_up":
                    err = safe_press("volumeup"); action_fired = "vol_up" if err is None else err
                else:
                    err = safe_press("volumedown"); action_fired = "vol_down" if err is None else err
                last_vol_time = now

        if action_fired is not None: last_hud_action = action_fired

        # periodic terminal print
        if (now - last_print) >= print_interval:
            ctext = f"{conf:.2f}" if conf is not None else "-"
            print(f"[pred:{cur or '-'} conf:{ctext}]  "
                  f"[stable:{stable_label or '-'} | ep:{episode_id}]  "
                  f"[theta:{theta_rad:+.2f} d:{dtheta_rad:+.2f} acc:{theta_accum_rad:+.2f}]  "
                  f"[backend:{'mp' if use_mp else 'cv'}] [last:{last_hud_action}]")
            last_print = now

        # HUD overlay
        if CFG["SHOW_HUD"] == 1:
            ctext = f"{conf:.2f}" if (conf is not None and cur is not None) else "-"
            tag2 = f"{tag}+ROI" if dyn_roi_mask is not None else tag
            put(vis, 10, 20, f"mask:{tag2} backend:{'mp' if use_mp else 'cv'}")
            put(vis, 10, 42, f"pred:{cur or '(none)'} conf:{ctext}")
            put(vis, 10, 64, f"theta:{theta_rad:+.2f} d:{dtheta_rad:+.2f} acc:{theta_accum_rad:+.2f}")
            put(vis, 10, 86, f"stable:{stable_label or '(none)'} | last:{last_hud_action}")
            cv2.imshow("gesture controller (CSV19 + dynROI + radians-rot)", vis)
        else:
            cv2.imshow("gesture controller (CSV19 + dynROI + radians-rot)", vis)

        # hotkeys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            counts = {k:0.0 for k in counts}
            label_hist.clear()
            stable_label=None; first_seen_time=0.0
            episode_id += 1; last_hud_action="(reset)"
            trail_pts.clear()
            if trail_canvas is not None: trail_canvas[:] = 0
            rot_center = None; prev_theta = None; theta_accum_rad = 0.0
            theta_rad = 0.0; dtheta_rad = 0.0
            print("[reset] state cleared")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
