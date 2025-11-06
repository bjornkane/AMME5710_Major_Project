# media_controller_mask_v8_model_infer_stable_terminal_v2.py
#
# EXACT collector v8 masking + ML inference
# - Stable episode-based logic ONLY for play/pause/seek/next/prev (no glitch)
# - Volume uses IMMEDIATE "gesture is currently shown" logic:
#       if majority says 'vol_up' or 'vol_down' right now → press at interval
#       if not in frame / different gesture → no volume keypresses
# - Terminal HUD prints prediction, stable label, live volume vote, and media state
#
# Keys: q = quit, r = reset
#
import argparse, time, os, sys
import numpy as np
import cv2
from collections import deque

import pyautogui
import joblib

# ------------------ Static thresholds (collector v8) ------------------
HSV1_LO = np.array([  0,  20,  40], dtype=np.uint8)
HSV1_HI = np.array([ 25, 255, 255], dtype=np.uint8)
HSV2_LO = np.array([170,  15,  40], dtype=np.uint8)
HSV2_HI = np.array([179, 255, 255], dtype=np.uint8)

# Y,Cr,Cb widened
YCC_LO = np.array([  0, 120,  80], dtype=np.uint8)
YCC_HI = np.array([255, 200, 140], dtype=np.uint8)

KERNEL       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

LABEL_FONT_SCALE = 0.5
LABEL_THICK      = 1

def put_kv(img, x, y, text, color=(40,255,40)):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE, color, LABEL_THICK, cv2.LINE_AA)

def masked_blend(bgr, mask_u8, color=(0,255,0), alpha=0.35):
    overlay = np.full_like(bgr, color, dtype=np.uint8)
    blended = cv2.addWeighted(bgr, 1.0, overlay, alpha, 0.0)
    out = bgr.copy()
    m = mask_u8 > 0
    out[m] = blended[m]
    return out

# ------------------ Preprocessing (exact as collector) ------------------
def gray_world_wb(bgr):
    b, g, r = cv2.split(bgr.astype(np.float32))
    mb, mg, mr = b.mean()+1e-6, g.mean()+1e-6, r.mean()+1e-6
    mgray = (mb + mg + mr) / 3.0
    b *= (mgray/mb); g *= (mgray/mg); r *= (mgray/mr)
    out = cv2.merge([b, g, r])
    return np.clip(out, 0, 255).astype(np.uint8)

def adaptive_gamma_from_Y(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:,:,0].astype(np.float32) / 255.0
    m = float(np.clip(Y.mean(), 1e-3, 0.999))
    gamma = np.clip(1.4 - m, 0.7, 1.3)
    inv = 1.0 / gamma
    lut = np.array([(i/255.0)**inv * 255.0 for i in range(256)], np.float32).clip(0,255).astype(np.uint8)
    return cv2.LUT(bgr, lut)

def preprocess_shared(raw_bgr):
    wb   = gray_world_wb(raw_bgr)
    adj  = adaptive_gamma_from_Y(wb)
    blur = cv2.GaussianBlur(adj, (5, 5), 0)
    return blur

# ------------------ Masks (exact as collector) ------------------
def postprocess_mask(mask_raw):
    mask = (mask_raw > 0).astype(np.uint8) * 255
    mask = cv2.erode(mask, KERNEL, iterations=1)
    mask = cv2.dilate(mask, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE, iterations=1)
    return mask

def apply_roi(mask, frac=0.45, side="none"):
    if frac <= 0.0 or side == "none":
        return mask
    h, w = mask.shape[:2]
    x_cut = int(w * frac)
    roi_mask = np.zeros_like(mask)
    if side == "right":
        roi_mask[:, x_cut:] = 255
    elif side == "left":
        roi_mask[:, : w - x_cut] = 255
    else:
        return mask
    return cv2.bitwise_and(mask, roi_mask)

def build_masks_shared(raw_bgr, roi_frac=0.45, roi_side="none"):
    blur = preprocess_shared(raw_bgr)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, HSV1_LO, HSV1_HI)
    m2  = cv2.inRange(hsv, HSV2_LO, HSV2_HI)
    mask_hsv = postprocess_mask(cv2.bitwise_or(m1, m2))

    ycc      = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    mask_ycc = postprocess_mask(cv2.inRange(ycc, YCC_LO, YCC_HI))

    mask_hsv = apply_roi(mask_hsv, roi_frac, roi_side)
    mask_ycc = apply_roi(mask_ycc, roi_frac, roi_side)
    return mask_hsv, mask_ycc

def largest_blob(mask_bin, min_area=2000, max_area_abs=np.inf):
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    valid = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= min_area and a <= max_area_abs:
            valid.append(c)
    if not valid: return None
    return max(valid, key=cv2.contourArea)

def mask_score(mask_bin, frame_shape):
    h, w = frame_shape[:2]
    frame_area = float(h*w)
    cnt = largest_blob(mask_bin, min_area=1200, max_area_abs=0.6*frame_area)
    if cnt is None: return 0.0, None
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = max(cv2.contourArea(hull), 1e-6)
    solidity = area / hull_area
    peri = max(cv2.arcLength(cnt, True), 1e-6)
    circularity = (4.0*np.pi*area)/(peri*peri)
    circularity = float(np.clip(circularity, 0.2, 1.0))
    area_ratio = area / frame_area
    score = float(area_ratio * solidity * circularity)
    return score, cnt

def auto_fuse_hsv_ycc(hsv_pp, ycc_pp, frame_shape, prefer='and'):
    and_mask = cv2.bitwise_and(hsv_pp, ycc_pp)
    or_mask  = cv2.bitwise_or(hsv_pp,  ycc_pp)

    s_hsv, c_hsv = mask_score(hsv_pp, frame_shape)
    s_ycc, c_ycc = mask_score(ycc_pp, frame_shape)
    s_and, c_and = mask_score(and_mask, frame_shape)
    s_or,  c_or  = mask_score(or_mask,  frame_shape)

    AND_MIN    = 0.0006
    SINGLE_MIN = 0.0008

    if prefer == 'and' and s_and >= AND_MIN:
        return and_mask, "AND", c_and
    if s_hsv >= SINGLE_MIN or s_ycc >= SINGLE_MIN:
        if s_hsv >= 1.15 * s_ycc: return hsv_pp, "HSV", c_hsv
        if s_ycc >= 1.15 * s_hsv: return ycc_pp, "YCrCb", c_ycc
    return or_mask, "OR", c_or

# ------------------ Geometry + features (matches collector CSV) ------------------
def contour_shape_features(cnt, frame_shape):
    area = cv2.contourArea(cnt)
    h, w = frame_shape[:2]
    frame_area = float(h*w)
    area_norm = area / frame_area if frame_area > 0 else 0.0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = (area / max(hull_area, 1e-6))
    x, y, bw, bh = cv2.boundingRect(cnt)
    aspect_ratio = (bw / max(bh, 1))
    peri = max(cv2.arcLength(cnt, True), 1e-6)
    circularity = (4.0*np.pi*area/(peri*peri))
    return area_norm, solidity, aspect_ratio, circularity

def get_palm_center_and_angle(cnt):
    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-6:
        cx, cy = 0.0, 0.0
    else:
        cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
    angle_rad = 0.0
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        angle_rad = (np.deg2rad(ellipse[2]) + np.pi) % (2*np.pi) - np.pi
    return (cx, cy), angle_rad

def get_fingertip_candidates(cnt, palm_c):
    hull = cv2.convexHull(cnt, returnPoints=True)
    pts  = hull.reshape(-1, 2).astype(np.float32)
    pcx, pcy = palm_c
    dists = np.linalg.norm(pts - np.array([pcx, pcy], np.float32), axis=1)
    idx_sorted = np.argsort(-dists)
    tips, used = [], []
    for idx in idx_sorted:
        x,y = pts[idx]; ang = np.arctan2(y-pcy, x-pcx)
        if any(abs(((ang-a+np.pi)%(2*np.pi))-np.pi) < np.deg2rad(15) for a in used):
            continue
        tips.append((x,y,dists[idx],ang)); used.append(ang)
        if len(tips) >= 5: break
    while len(tips) < 5: tips.append((pcx,pcy,0.0,0.0))
    return sorted(tips, key=lambda t: t[3]), len(used)

def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def compute_motion_features(trail_pts, frame_shape):
    h, w = frame_shape[:2]; diag = np.sqrt(w*w + h*h)
    if len(trail_pts) < 2:
        return dict(dx_norm=0.0, dy_norm=0.0, speed_norm=0.0, dir_angle=0.0, path_len_norm=0.0)
    (x0,y0) = trail_pts[-2]; (x1,y1) = trail_pts[-1]
    dx, dy = x1-x0, y1-y0
    speed_norm = np.hypot(dx,dy)/max(diag,1e-6)
    lookback = max(0, len(trail_pts)-6)
    (xo,yo) = trail_pts[lookback]
    dxL, dyL = x1-xo, y1-yo
    path = 0.0
    for i in range(1, len(trail_pts)):
        ax, ay = trail_pts[i-1]; bx, by = trail_pts[i]
        path += np.hypot(bx-ax, by-ay)
    return dict(dx_norm=float(dxL/max(diag,1e-6)),
                dy_norm=float(dyL/max(diag,1e-6)),
                speed_norm=float(speed_norm),
                dir_angle=float(np.arctan2(dyL, dxL)),
                path_len_norm=float(path/max(diag,1e-6)))

def build_feature_vector(
    palm_c, palm_angle, tips_sorted, num_real_tips, frame_shape,
    prev_palm_angle, prev_time, now_time,
    area_norm, solidity, aspect_ratio, circularity,
    motion_feat
):
    h, w = frame_shape[:2]
    pcx, pcy = palm_c
    pcx_n = pcx/float(w) if w>0 else 0.0
    pcy_n = pcy/float(h) if h>0 else 0.0
    dists = [t[2] for t in tips_sorted]
    hand_scale = max(dists) if dists else 1.0
    if hand_scale < 1e-6: hand_scale = 1.0
    diag = np.sqrt(w*w + h*h)
    hand_scale_n = (hand_scale/max(diag,1e-6))

    ft_feat, rels = [], []
    for (_, _, dist, ang) in tips_sorted:
        d_n = dist/hand_scale
        a_rel = wrap_angle(ang - palm_angle)
        ft_feat.extend([d_n, a_rel]); rels.append(a_rel)

    if prev_palm_angle is not None and prev_time is not None:
        dtheta = wrap_angle(palm_angle - prev_palm_angle)
        dt = now_time - prev_time
        ang_vel = dtheta/dt if dt>0 else 0.0
    else:
        ang_vel = 0.0

    angular_spread = abs(wrap_angle(max(rels)-min(rels))) if rels else 0.0
    thumb_angle_rel = wrap_angle(tips_sorted[int(np.argmax(dists))][3]-palm_angle) if dists else 0.0

    feats = [
        pcx_n, pcy_n, palm_angle
    ] + ft_feat + [
        ang_vel, area_norm, solidity, aspect_ratio, circularity,
        float(num_real_tips), angular_spread, thumb_angle_rel, hand_scale_n,
        motion_feat["dx_norm"], motion_feat["dy_norm"], motion_feat["speed_norm"],
        motion_feat["dir_angle"], motion_feat["path_len_norm"]
    ]
    return np.array(feats, dtype=np.float32), ang_vel

# ------------------ Face ignore (optional) ------------------
def load_face_detector():
    xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(xml)

def face_near_hand(face_boxes, hand_bbox, iou_thresh=0.15):
    if hand_bbox is None: return False
    x,y,w,h = hand_bbox
    hx1, hy1, hx2, hy2 = x, y, x+w, y+h
    for (fx, fy, fw, fh) in face_boxes:
        fx1, fy1, fx2, fy2 = fx, fy, fx+fw, fy+fh
        ix1, iy1 = max(hx1, fx1), max(hy1, fy1)
        ix2, iy2 = min(hx2, fx2), min(hy2, fy2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw*ih
        union = (w*h) + (fw*fh) - inter + 1e-6
        if inter/union >= iou_thresh:
            return True
    return False

# ------------------ Media actions ------------------
def safe_press(key):
    try:
        pyautogui.press(key)
        return None
    except Exception as e:
        return f"pyautogui error: {e}"

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser(description="v8 mask + model inference (stable play/pause; live volume) + terminal HUD")
    # camera / size / perf
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--prefer", choices=["and","hsv","ycc"], default="and")
    ap.add_argument("--roi_side", choices=["none","right","left"], default="none")
    ap.add_argument("--roi_frac", type=float, default=0.45)
    ap.add_argument("--min_area", type=int, default=1800)
    ap.add_argument("--max_area_frac", type=float, default=0.5)
    ap.add_argument("--hud", type=int, default=1)

    # model
    ap.add_argument("--gesture_scaler", default="gesture_scaler.pkl")
    ap.add_argument("--model",  default="gesture_model.pkl")
    ap.add_argument("--labels", default="label_map.npy")

    # stability (discrete)
    ap.add_argument("--hist_len", type=int, default=7)
    ap.add_argument("--majority", type=int, default=4)
    ap.add_argument("--hold_ms", type=int, default=250)
    ap.add_argument("--trail_len", type=int, default=20)
    ap.add_argument("--smooth_alpha", type=float, default=0.4)

    # cooldowns
    ap.add_argument("--global_cooldown", type=float, default=0.35)

    # volume (live)
    ap.add_argument("--vol_interval", type=float, default=0.12)
    ap.add_argument("--vol_hist_len", type=int, default=3)
    ap.add_argument("--vol_majority", type=int, default=2)

    # face ignore
    ap.add_argument("--face", type=int, default=0)
    ap.add_argument("--face_scale", type=float, default=1.15)
    ap.add_argument("--face_neighbors", type=int, default=5)

    # terminal print throttle
    ap.add_argument("--print_hz", type=float, default=4.0)

    args = ap.parse_args()

    cv2.setUseOptimized(True)
    try:
        pyautogui.FAILSAFE = False
    except Exception:
        pass

    # load model parts
    try:
        gesture_scaler = joblib.load(args.gesture_scaler)
    except Exception as e:
        print(f"[ERROR] Failed to load gesture_scaler at '{args.gesture_scaler}': {e}")
        return

    try:
        clf = joblib.load(args.model)
    except Exception as e:
        print(f"[ERROR] Failed to load model at '{args.model}': {e}")
        return

    try:
        label_map = np.load(args.labels, allow_pickle=True)
    except Exception:
        label_map = None

    face_cascade = load_face_detector() if args.face == 1 else None

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if args.width > 0:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    if args.height > 0: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print("ERROR: camera not available. Try --cam 1 or close other apps.")
        return

    # state
    label_hist = deque(maxlen=max(3, args.hist_len))      # for discrete stable actions
    vol_hist   = deque(maxlen=max(2, args.vol_hist_len))  # for live volume
    stable_label = None
    first_seen_time = 0.0

    episode_id = 0
    last_fired_episode_for_label = {}  # (label, episode_id) -> True

    media_state = "unknown"  # "unknown" | "playing" | "paused"

    last_action_time = 0.0
    last_vol_time = 0.0
    last_hud_action = "(idle)"

    trail = deque(maxlen=max(8, args.trail_len))
    last_tip = None
    last_palm_angle = None
    last_time = None

    last_print = 0.0
    print_interval = 1.0 / max(args.print_hz, 0.1)

    try:
        while True:
            ok, frame_raw = cap.read()
            if not ok: break
            raw = cv2.flip(frame_raw, 1)
            h, w = raw.shape[:2]
            frame_area = float(h*w)

            # exact masks
            mask_hsv_pp, mask_ycc_pp = build_masks_shared(raw, args.roi_frac, args.roi_side)
            mask_fused, fuse_tag, _ = auto_fuse_hsv_ycc(mask_hsv_pp, mask_ycc_pp, raw.shape, prefer=args.prefer)

            # keep largest blob only
            cnt_keep = largest_blob(mask_fused, min_area=args.min_area, max_area_abs=args.max_area_frac*frame_area)
            mask_final = np.zeros_like(mask_fused)
            if cnt_keep is not None:
                cv2.drawContours(mask_final, [cnt_keep], -1, 255, -1)

            # optional face ignore
            face_block = False
            faces = []
            if args.face == 1:
                g = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(g, scaleFactor=args.face_scale, minNeighbors=args.face_neighbors)

            vis = raw if args.hud == 0 else masked_blend(raw, mask_final, color=(0,255,0), alpha=0.35)

            now = time.time()
            cur_pred = None

            if cnt_keep is not None:
                x,y,bw,bh = cv2.boundingRect(cnt_keep)
                if args.face == 1:
                    face_block = face_near_hand(faces, (x,y,bw,bh))

                (pcx, pcy), palm_angle = get_palm_center_and_angle(cnt_keep)
                tips_sorted, num_real_tips = get_fingertip_candidates(cnt_keep, (pcx, pcy))
                area_norm, solidity, aspect_ratio, circularity = contour_shape_features(cnt_keep, raw.shape)

                # leader fingertip for motion features
                donly = [t[2] for t in tips_sorted]
                lead_i = int(np.argmax(donly)) if donly else 0
                if tips_sorted:
                    tx, ty = tips_sorted[lead_i][0], tips_sorted[lead_i][1]
                    if last_tip is None:
                        sx, sy = tx, ty
                    else:
                        px, py = last_tip
                        a = float(np.clip(args.smooth_alpha, 0.0, 1.0))
                        sx = a*tx + (1.0-a)*px
                        sy = a*ty + (1.0-a)*py
                    last_tip = (sx, sy)
                    trail.append((sx, sy))

                motion_feat = compute_motion_features(trail, raw.shape)
                feats, _ = build_feature_vector(
                    (pcx, pcy), palm_angle, tips_sorted, num_real_tips, raw.shape,
                    last_palm_angle, last_time, now,
                    area_norm, solidity, aspect_ratio, circularity,
                    motion_feat
                )
                last_palm_angle = palm_angle
                last_time = now

                # predict
                try:
                    X = gesture_scaler.transform(feats.reshape(1,-1))
                    pred = clf.predict(X)[0]
                    if label_map is not None and isinstance(pred, (int, np.integer)) and pred < len(label_map):
                        cur_pred = str(label_map[pred])
                    else:
                        cur_pred = str(pred)
                except Exception as e:
                    cur_pred = None
                    if args.hud == 1:
                        put_kv(vis, 10, 24, f"model error: {e}", (0,0,255))

                if args.hud == 1:
                    cv2.drawContours(vis, [cnt_keep], -1, (0,255,255), 2)
                    hull = cv2.convexHull(cnt_keep)
                    cv2.polylines(vis, [hull], isClosed=True, color=(255,100,0), thickness=2)
                    cv2.circle(vis, (int(pcx), int(pcy)), 5, (0,0,255), -1)
                    L = 50
                    x2 = int(pcx + L*np.cos(palm_angle)); y2 = int(pcy + L*np.sin(palm_angle))
                    cv2.line(vis, (int(pcx), int(pcy)), (x2, y2), (0,0,255), 2)
                    for i,(tx,ty,_,_) in enumerate(tips_sorted):
                        cv2.circle(vis, (int(tx), int(ty)), 5, (255,0,0), 2)
            else:
                trail.clear()
                last_tip = None
                cur_pred = None

            # update histories
            if cur_pred is not None:
                label_hist.append(cur_pred)
                vol_hist.append(cur_pred)

            # discrete stable label (play/pause/seek/next/prev)
            action_fired = None
            if not face_block:
                # majority over label_hist
                top = None
                best = 0
                for s in set(label_hist):
                    c = sum(1 for v in label_hist if v == s)
                    if c > best:
                        best = c; top = s
                if top is not None and best >= args.majority:
                    if stable_label != top:
                        stable_label = top
                        first_seen_time = now
                        episode_id += 1
                else:
                    if stable_label is not None:
                        stable_label = None
                        episode_id += 1
                    first_seen_time = now

                # fire discrete when stable long enough
                if stable_label is not None:
                    held_ms = (now - first_seen_time) * 1000.0
                    if held_ms >= args.hold_ms:
                        can_global = (now - last_action_time) > float(args.global_cooldown)
                        if stable_label == "play" and can_global:
                            if media_state != "playing" and last_fired_episode_for_label.get(("play", episode_id)) is None:
                                err = safe_press("playpause")
                                if err is None:
                                    media_state = "playing"
                                    action_fired = "play"
                                    last_action_time = now
                                    last_fired_episode_for_label[("play", episode_id)] = True
                                else:
                                    action_fired = err

                        elif stable_label == "pause" and can_global:
                            if media_state != "paused" and last_fired_episode_for_label.get(("pause", episode_id)) is None:
                                err = safe_press("playpause")
                                if err is None:
                                    media_state = "paused"
                                    action_fired = "pause"
                                    last_action_time = now
                                    last_fired_episode_for_label[("pause", episode_id)] = True
                                else:
                                    action_fired = err

                        elif stable_label == "next" and can_global:
                            if last_fired_episode_for_label.get(("next", episode_id)) is None:
                                err = safe_press("nexttrack")
                                action_fired = "next" if err is None else err
                                last_action_time = now
                                last_fired_episode_for_label[("next", episode_id)] = True

                        elif stable_label == "prev" and can_global:
                            if last_fired_episode_for_label.get(("prev", episode_id)) is None:
                                err = safe_press("prevtrack")
                                action_fired = "prev" if err is None else err
                                last_action_time = now
                                last_fired_episode_for_label[("prev", episode_id)] = True

                        elif stable_label in ("seek_right", "forward") and can_global:
                            if last_fired_episode_for_label.get(("seek_right", episode_id)) is None:
                                err = safe_press("right")
                                action_fired = "seek_right" if err is None else err
                                last_action_time = now
                                last_fired_episode_for_label[("seek_right", episode_id)] = True

                        elif stable_label in ("seek_left", "rewind") and can_global:
                            if last_fired_episode_for_label.get(("seek_left", episode_id)) is None:
                                err = safe_press("left")
                                action_fired = "seek_left" if err is None else err
                                last_action_time = now
                                last_fired_episode_for_label[("seek_left", episode_id)] = True

            # live volume control (separate majority window; no hold_ms; stops automatically)
            vol_vote = None
            if not face_block and cnt_keep is not None:
                # majority over vol_hist
                if len(vol_hist) > 0:
                    counts = {}
                    for v in vol_hist:
                        counts[v] = counts.get(v, 0) + 1
                    vol_vote = max(counts, key=counts.get)
                    if counts[vol_vote] < args.vol_majority:
                        vol_vote = None

                now2 = time.time()
                if vol_vote == "vol_up" and (now2 - last_vol_time) >= float(args.vol_interval):
                    err = safe_press("volumeup")
                    action_fired = "vol_up (hold)" if err is None else err
                    last_vol_time = now2
                elif vol_vote == "vol_down" and (now2 - last_vol_time) >= float(args.vol_interval):
                    err = safe_press("volumedown")
                    action_fired = "vol_down (hold)" if err is None else err
                    last_vol_time = now2
            else:
                vol_vote = None  # no hand / blocked → no volume output

            if action_fired is not None:
                last_hud_action = action_fired

            # terminal HUD (throttled)
            if (now - last_print) >= print_interval:
                print(f"[pred:{cur_pred or '-'}]  [stable:{stable_label or '-'} | ep:{episode_id}]  "
                      f"[vol_vote:{vol_vote or '-'}]  [media:{media_state}]  [last:{last_hud_action}]  "
                      f"[face_block:{face_block}]")
                last_print = now

            # on-screen HUD
            if args.hud == 1:
                put_kv(vis, 10, 20, f"mask:{fuse_tag} | ROI:{args.roi_side}")
                put_kv(vis, 10, 42, f"pred:{cur_pred if cur_pred else '(none)'} | stable:{stable_label if stable_label else '(none)'} | ep:{episode_id}")
                put_kv(vis, 10, 64, f"vol_vote:{vol_vote if vol_vote else '(none)'} | media:{media_state} | last:{last_hud_action}")
                if args.face == 1:
                    for (fx,fy,fw,fh) in faces:
                        cv2.rectangle(vis, (fx,fy), (fx+fw,fy+fh), (0,180,255), 2)
                    if face_block:
                        put_kv(vis, 10, 86, "face near hand → IGNORE", (0,200,255))
                cv2.imshow("v8 mask + model inference (stable discrete, live volume)", vis)
            else:
                cv2.imshow("v8 mask + model inference", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                label_hist.clear()
                vol_hist.clear()
                trail.clear()
                last_tip = None
                stable_label = None
                first_seen_time = 0.0
                episode_id += 1
                last_hud_action = "(reset)"
                print("[reset] state cleared")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
