# collect_data.py
# 
# AMME5710 - Computer Vision and Image Processing - Major Project
# Authors: Varunvarshan Sideshkumar, Arin Adurkar, Siwon Kang
# Purpose of this code:
#   - Collect exactly N training samples per run for a single label.
#   - Save raw frames, hand-only masks, cropped ROIs, and overlay images.
#   - Log geometric features + simple radians-based rotation features to CSV.
#   - Show a compact 2x3 viewer:
#       [ RAW | HSV mask | YCrCb mask ]
#       [ FINAL hand-only | FEED + Contour | RAW + TRAIL ]
#
# Usage:
#   python collect_data.py --label vol_up --frames 100

import argparse, os, csv, time, math
from collections import deque
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

# Camera / UI config
CAM_INDEX     = 0
FRAME_W       = 960
FRAME_H       = 540
FPS_TARGET    = 30

CELL_H        = 220                   # tile height in the 2x3 viewer
Y_GATE_LO     = 0.30                  # keep rows from 30% to bottom (avoid face)
Y_GATE_HI     = 1.00
TRAIL_DECAY   = 0.92                  # fade for motion trail buffer
TRAIL_THICK   = 3
TRAIL_MIN_NRM = 0.002                 # min normalized motion to draw a trail segment

# radians (cw/ccw) config
RAD_PIVOT_ALPHA   = 0.15              # EMA for pivot center
RAD_MIN_RADIUS_N  = 0.06              # min radius as a fraction of image diagonal
RAD_DTH_SMOOTH    = 0.35              # EMA smoothing for dtheta
RAD_ACCUM_DECAY   = 0.95              # decay for angle accumulator
RAD_VOTE_THRESH   = 0.20              # |accumulated angle| to vote cw/ccw
RAD_TEXT_Y        = 84                # HUD baseline for radians text

# colour thresholds + kernels
HSV1_LO = np.array([  0,  25,  45], np.uint8)
HSV1_HI = np.array([ 25, 255, 255], np.uint8)
HSV2_LO = np.array([170,  20,  45], np.uint8)
HSV2_HI = np.array([179, 255, 255], np.uint8)
YCC_LO  = np.array([  0, 133,  77], np.uint8)
YCC_HI  = np.array([255, 173, 127], np.uint8)

KERNEL       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
LABEL_FONT_SCALE = 0.5
LABEL_THICK      = 1

TUNE_WIN = "Tune HSV/YCrCb"
GRID_WIN = "layout: [RAW | HSV | YCrCb] / [FINAL | FEED+PRIOR | RAW⊙TRAIL]"

# CLI
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label",  type=str, required=True)  # class name for this run
    ap.add_argument("--frames", type=int, required=True)  # number of samples to save
    return ap.parse_args()

# I/O setup
def ensure_paths(label: str):
    # per label folders for this collection
    root = Path("data") / f"data_{label}"
    (root / "images_raw").mkdir(parents=True, exist_ok=True)
    (root / "images_crop").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    (root / "overlays").mkdir(parents=True, exist_ok=True)

    # global CSV for all labels
    csv_path = Path("data") / "gestures_log.csv"
    new_csv = not csv_path.exists()
    fh = open(csv_path, "a", newline="")
    wr = csv.writer(fh)
    if new_csv:
        wr.writerow([
            "image_id","time_s","label",
            "w","h","cx","cy","cx_n","cy_n",
            "area","area_norm","solidity","aspect_ratio","circularity",
            "bbox_x","bbox_y","bbox_w","bbox_h","extent",
            "mask_fuse","finger_count","prior_nboxes",
            "theta_rad","dtheta_rad","theta_accum_rad","rot_dir",
            "path_raw","path_crop","path_mask","path_overlay"
        ])
    return root, fh, wr

# small drawing helpers
def put_kv(img, x, y, text):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE, (40, 255, 40), LABEL_THICK, cv2.LINE_AA)

def tile2x3(a,b,c,d,e,f, cell_h=CELL_H, pad=6, bg=(20,20,20)):
    # put six panels into a neat 2x3 grid for live preview
    def rz(x, h=cell_h):
        h0, w0 = x.shape[:2]
        if h0 == h: return x
        w = int(round(w0 * (h/float(h0))))
        return cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA if h < h0 else cv2.INTER_LINEAR)
    panels = [rz(a), rz(b), rz(c), rz(d), rz(e), rz(f)]
    colw = [max(p.shape[1] for p in panels[i::3]) for i in range(3)]
    canvas = np.full((cell_h*2 + pad*3, sum(colw) + pad*4, 3), bg, np.uint8)
    y = pad; idx = 0
    for _ in range(2):
        x = pad
        for ci in range(3):
            p = panels[idx]; h, w = p.shape[:2]
            y0 = y + (cell_h-h)//2; x0 = x + (colw[ci]-w)//2
            canvas[y0:y0+h, x0:x0+w] = p
            x += colw[ci] + pad
            idx += 1
        y += cell_h + pad
    return canvas

# colour preprocessing
def gray_world_wb(bgr):
    # simple gray-world white balance
    b,g,r = cv2.split(bgr.astype(np.float32))
    mb,mg,mr = b.mean()+1e-6, g.mean()+1e-6, r.mean()+1e-6
    m = (mb+mg+mr)/3.0
    out = cv2.merge([b*(m/mb), g*(m/mg), r*(m/mr)])
    return np.clip(out,0,255).astype(np.uint8)

def adaptive_gamma_from_Y(bgr):
    # lighten dark scenes / tame bright ones based on Y channel mean
    Y = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)[:,:,0].astype(np.float32)/255.0
    gamma = np.clip(1.4 - float(np.clip(Y.mean(),1e-3,0.999)), 0.7, 1.3)
    inv = 1.0/gamma
    lut = np.array([(i/255.0)**inv * 255.0 for i in range(256)], np.uint8)
    return cv2.LUT(bgr, lut)

def postprocess_mask(m):
    # clean edges and fill small gaps
    m = (m>0).astype(np.uint8)*255
    m = cv2.erode(m, KERNEL, 1)
    m = cv2.dilate(m, KERNEL, 2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, KERNEL_CLOSE, 1)
    return m

def apply_vertical_gate(mask, y_lo_frac=Y_GATE_LO, y_hi_frac=Y_GATE_HI):
    # ignore top band to reduce face interference
    h,w = mask.shape[:2]
    y0 = int(np.clip(y_lo_frac,0,1)*h); y1 = int(np.clip(y_hi_frac,0,1)*h)
    gate = np.zeros_like(mask); gate[y0:y1,:] = 255
    return cv2.bitwise_and(mask, gate)

def build_masks(raw_bgr):
    # produce HSV and YCrCb masks with the same preprocessing
    wb   = gray_world_wb(raw_bgr)
    adj  = adaptive_gamma_from_Y(wb)
    blur = cv2.GaussianBlur(adj,(5,5),0)

    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    m1   = cv2.inRange(hsv, HSV1_LO, HSV1_HI)
    m2   = cv2.inRange(hsv, HSV2_LO, HSV2_HI)
    m_h  = postprocess_mask(cv2.bitwise_or(m1,m2))

    ycc  = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    m_y  = postprocess_mask(cv2.inRange(ycc, YCC_LO, YCC_HI))

    m_h  = apply_vertical_gate(m_h)
    m_y  = apply_vertical_gate(m_y)
    return m_h, m_y

# simple mask scoring + fusion
def largest_blob(m, min_area=2000, max_frac=0.6, shape=None):
    # return largest contour under simple area limits
    h,w = m.shape[:2]; max_area = (h*w if shape is None else shape[0]*shape[1])*max_frac
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_a = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a>min_area and a<max_area and a>best_a:
            best, best_a = c, a
    return best

def mask_score(m):
    # estimate mask quality using area, convexity, and circularity
    cnt = largest_blob(m)
    if cnt is None: return 0.0
    a = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    s = a/max(cv2.contourArea(hull),1e-6)
    p = max(cv2.arcLength(cnt,True),1e-6)
    c = (4.0*np.pi*a)/(p*p)
    return float((a*m.size**-1) * s * np.clip(c,0.2,1.0))

def iou(a,b):
    inter = np.count_nonzero(cv2.bitwise_and(a,b))
    uni   = max(1, np.count_nonzero(cv2.bitwise_or(a,b)))
    return inter/uni

def fuse(hsv_m, ycc_m, shape):
    # AND if both agree and are strong; else pick stronger one; else OR as fallback
    and_m = cv2.bitwise_and(hsv_m, ycc_m)
    or_m  = cv2.bitwise_or(hsv_m, ycc_m)
    s_h, s_y, s_and = mask_score(hsv_m), mask_score(ycc_m), mask_score(and_m)
    if iou(hsv_m,ycc_m) >= 0.35 and s_and >= 5e-4:
        return and_m, "AND"
    if s_h >= 7e-4 or s_y >= 7e-4:
        if s_h >= 1.15*s_y: return hsv_m, "HSV"
        if s_y >= 1.15*s_h: return ycc_m, "YCrCb"
        return hsv_m, "HSV"
    return or_m, "OR"

def select_hand_like(m, shape):
    # pick a blob that looks hand-like (area/convexity/circularity/extent heuristics)
    h,w = shape[:2]
    cnts,_ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 2000 or a > 0.18*h*w: continue
        hull = cv2.convexHull(c); s = a/max(cv2.contourArea(hull),1e-6)
        p = max(cv2.arcLength(c,True),1e-6); circ = (4.0*np.pi*a)/(p*p)
        if circ > 0.88: continue  # too round -> likely noise
        x,y,bw,bh = cv2.boundingRect(c); extent = a/max(1,bw*bh)
        score = (a/(h*w))**0.9 * s**0.6 * (1.0-circ)**1.2 * extent**0.4
        if score > best_score: best, best_score = c, score
    return best

def safe_convexity_defects(cnt):
    # convexity defects can fail if hull/contour is degenerate -> try approx
    if cnt is None or len(cnt)<3: return None, None
    c = cnt.astype(np.int32)
    def try_def(c0):
        try:
            h = cv2.convexHull(c0, returnPoints=False)
            if h is None or len(h)<3: return None
            return cv2.convexityDefects(c0, h), c0
        except cv2.error:
            return None
    out = try_def(c)
    if out: return out
    approx = cv2.approxPolyDP(c, 2.0, True)
    out = try_def(approx)
    return out if out else (None, None)

def gesture_from_cnt(cnt):
    # quick label for HUD only (Fist/Open/Unknown + finger count)
    if cnt is None: return "No Hand",  -1
    defects, used = safe_convexity_defects(cnt)
    if defects is None:
        a = cv2.contourArea(cnt); p = max(cv2.arcLength(cnt,True),1e-6)
        circ = (4.0*np.pi*a)/(p*p)
        return ("Fist" if circ>0.85 else "Unknown"), (0 if circ>0.85 else -1)
    c = used if used is not None else cnt
    fingers = 0
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        A = np.linalg.norm(c[e][0]-c[s][0])
        B = np.linalg.norm(c[f][0]-c[s][0])
        C = np.linalg.norm(c[e][0]-c[f][0])
        if B<=1e-6 or C<=1e-6: continue
        ang = np.arccos(np.clip((B*B + C*C - A*A)/(2*B*C), -1, 1))
        if ang <= np.pi/2 and d > 2000: fingers += 1
    mapping = {0:"Fist",1:"Point",2:"Peace",3:"Three",4:"Four"}
    return (mapping.get(fingers,"Open Hand" if fingers>=5 else "Unknown"),
            fingers if fingers<=4 else 5)

# MediaPipe prior (for gating)
class MPPrior:
    # tries to get rough hand boxes / polygons to gate masks
    def __init__(self):
        self.ok = False
        self.last_polys: List[np.ndarray] = []
        try:
            import mediapipe as mp
            self.h = mp.solutions.hands.Hands(False, max_num_hands=2,
                                              min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5)
            self.ok = True
        except Exception:
            self.ok = False
    def infer(self, bgr):
        self.last_polys = []
        if not self.ok: return []
        H,W = bgr.shape[:2]
        out = []
        res = self.h.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if res and res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                xs = [int(p.x*W) for p in lm.landmark]
                ys = [int(p.y*H) for p in lm.landmark]
                x1,y1,x2,y2 = max(0,min(xs)), max(0,min(ys)), min(W,max(xs)), min(H,max(ys))
                out.append((x1,y1,x2,y2,"hand",0.9))
                hull = cv2.convexHull(np.array(list(zip(xs,ys)), np.int32))
                self.last_polys.append(hull.reshape(-1,1,2))
        return out

def gate_with_polys(m, polys, inflate_px=8):
    # keep mask only inside MediaPipe hand polygons (dilated slightly)
    if not polys: return m
    h,w = m.shape[:2]
    g = np.zeros((h,w), np.uint8)
    for poly in polys:
        if poly is None or len(poly)<3: continue
        cv2.fillPoly(g, [poly], 255)
    if inflate_px>0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate_px*2+1, inflate_px*2+1))
        g = cv2.dilate(g, k, 1)
    return cv2.bitwise_and(m, g)

def draw_boxes(img, dets, color=(255,200,0)):
    # draw MP boxes (for visual feedback only)
    o = img.copy()
    for (x1,y1,x2,y2,lab,sc) in dets:
        cv2.rectangle(o,(x1,y1),(x2,y2),color,2)
        cv2.putText(o,f"{lab}:{sc:.2f}",(x1,max(15,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2,cv2.LINE_AA)
    return o

# radians helpers
def unwrap(prev, cur):
    # map angle difference into (-pi, pi]
    d = cur - prev
    while d <= -math.pi: d += 2*math.pi
    while d >   math.pi: d -= 2*math.pi
    return d

def diag_len(w, h): return math.hypot(w, h)

# main loop
def main():
    args = parse_args()
    label = args.label
    frames_target = max(1, int(args.frames))

    root, csv_fh, csv_wr = ensure_paths(label)
    p_raw = root / "images_raw"
    p_crop= root / "images_crop"
    p_mask= root / "masks"
    p_ovl = root / "overlays"

    cv2.namedWindow(GRID_WIN, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    if not cap.isOpened():
        raise RuntimeError("Camera open failed")

    mp = MPPrior()                         # MediaPipe prior (if available)
    trail_pts = deque(maxlen=20)           # centroid path
    trail_canvas = None                    # faded trail buffer
    saved = 0                              # saved samples counter
    prior_boxes = []                       # MP boxes for overlay

    # radians state
    pivot_ema: Tuple[float,float] = None   # EMA pivot center
    theta_prev = None                      # previous angle
    dtheta_ema = 0.0                       # smoothed angle delta
    theta_accum = 0.0                      # accumulated rotation

    try:
        while saved < frames_target:
            ok, fr = cap.read()
            if not ok: break
            frame = cv2.flip(fr, 1)
            H,W = frame.shape[:2]
            if trail_canvas is None:
                trail_canvas = np.zeros((H,W), np.uint8)

            # build masks and fuse
            m_hsv, m_ycc = build_masks(frame)
            m_fuse, tag  = fuse(m_hsv, m_ycc, frame.shape)

            # gate with MP hand polygons if present
            prior_boxes = mp.infer(frame) if mp.ok else []
            if getattr(mp, "last_polys", None):
                m_gate = gate_with_polys(m_fuse, mp.last_polys, inflate_px=max(8,int(0.08*max(H,W))))
            else:
                m_gate = m_fuse

            # select final contour and a simple HUD label
            cnt = select_hand_like(m_gate, frame.shape)
            gesture, fcnt = gesture_from_cnt(cnt)

            # geometric features
            if cnt is not None:
                area = cv2.contourArea(cnt)
                hull = cv2.convexHull(cnt); hull_a = max(cv2.contourArea(hull),1e-6)
                solidity = area/hull_a
                x,y,bw,bh = cv2.boundingRect(cnt)
                extent = area/max(1,bw*bh)
                peri = max(cv2.arcLength(cnt,True),1e-6)
                circ = (4.0*np.pi*area)/(peri*peri)
                M = cv2.moments(cnt)
                cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
                cx_n, cy_n = cx/W, cy/H
                aspect = bw/max(bh,1)
                area_norm = area/(W*H)
            else:
                solidity=aspect=circ=extent=area_norm=-1.0
                x=y=bw=bh=-1
                cx=cy=cx_n=cy_n=-1.0

            # final hand-only mask (contour fill)
            m_final = np.zeros_like(m_gate)
            if cnt is not None:
                cv2.drawContours(m_final,[cnt],-1,255,-1)

            # centroid trail drawing (with motion threshold)
            if cnt is not None and cx>=0 and cy>=0:
                trail_pts.append((cx,cy))
            trail_canvas = (trail_canvas.astype(np.float32)*TRAIL_DECAY).astype(np.uint8)
            if len(trail_pts)>=2:
                x0,y0 = trail_pts[-2]; x1,y1 = trail_pts[-1]
                if np.hypot(x1-x0,y1-y0)/diag_len(W,H) >= TRAIL_MIN_NRM:
                    cv2.line(trail_canvas,(int(x0),int(y0)),(int(x1),int(y1)),255,TRAIL_THICK)

            # radians CW/CCW estimation
            theta_rad = 0.0
            dtheta_rad = 0.0
            rot_dir = 0  # -1=cw, +1=ccw
            if cnt is not None and cx>=0 and cy>=0:
                if pivot_ema is None:
                    pivot_ema = (cx, cy)
                else:
                    px, py = pivot_ema
                    pivot_ema = (px*(1.0-RAD_PIVOT_ALPHA) + cx*RAD_PIVOT_ALPHA,
                                 py*(1.0-RAD_PIVOT_ALPHA) + cy*RAD_PIVOT_ALPHA)
                px, py = pivot_ema

                vx, vy = cx - px, cy - py
                r = math.hypot(vx, vy)
                if r >= RAD_MIN_RADIUS_N * diag_len(W,H):
                    # image coordinates: y grows downward -> atan2(vy, vx)
                    theta_rad = math.atan2(vy, vx)
                    if theta_prev is None:
                        theta_prev = theta_rad
                    d = unwrap(theta_prev, theta_rad)
                    theta_prev = theta_rad
                    # smooth Δθ and update accumulator
                    dtheta_ema = dtheta_ema*(1.0-RAD_DTH_SMOOTH) + d*RAD_DTH_SMOOTH
                    dtheta_rad = dtheta_ema
                    theta_accum = theta_accum*RAD_ACCUM_DECAY + dtheta_ema
                    if theta_accum >= RAD_VOTE_THRESH:
                        rot_dir = +1  # ccw (screen coords)
                        theta_accum = 0.0
                    elif theta_accum <= -RAD_VOTE_THRESH:
                        rot_dir = -1  # cw
                        theta_accum = 0.0
                else:
                    # too close to pivot: only decay accumulated angle
                    theta_accum *= RAD_ACCUM_DECAY
            else:
                # no hand: decay and reset angle memory
                theta_accum *= RAD_ACCUM_DECAY
                theta_prev = None

            # viewer panels
            p0 = frame.copy(); put_kv(p0,10,20,"RAW")
            p1 = cv2.cvtColor(m_hsv, cv2.COLOR_GRAY2BGR); put_kv(p1,10,20,"HSV mask")
            p2 = cv2.cvtColor(m_ycc, cv2.COLOR_GRAY2BGR); put_kv(p2,10,20,"YCrCb mask")
            p3 = cv2.cvtColor(m_final, cv2.COLOR_GRAY2BGR); put_kv(p3,10,20,"FINAL (hand-only)")

            ov = draw_boxes(frame, prior_boxes)
            if cnt is not None:
                cv2.drawContours(ov,[cnt],-1,(0,255,255),2)
                if cx>=0 and cy>=0:
                    cv2.circle(ov,(int(cx), int(cy)),5,(0,0,255),-1)
                if pivot_ema is not None:
                    cv2.circle(ov,(int(pivot_ema[0]), int(pivot_ema[1])),4,(255,120,0),-1)
                    cv2.line(ov,(int(pivot_ema[0]),int(pivot_ema[1])),(int(cx),int(cy)),(255,120,0),2)

            theta_deg = theta_rad*180.0/math.pi
            thetaA_deg = theta_accum*180.0/math.pi
            dir_txt = "CCW" if rot_dir==+1 else ("CW" if rot_dir==-1 else "—")
            put_kv(ov,10,20,f"FEED + contour  fuse:{tag}")
            put_kv(ov,10,50,f"Gesture:{gesture}")
            put_kv(ov,10,RAD_TEXT_Y,   f"theta(rad)={theta_rad:+.3f} ({theta_deg:+.1f}°)")
            put_kv(ov,10,RAD_TEXT_Y+22,f"dtheta≈{dtheta_ema:+.3f}  Θ_accum≈{theta_accum:+.3f}  dir:{dir_txt}")

            trail_rgb = np.zeros_like(frame); trail_rgb[:,:,1] = np.clip(trail_canvas,0,255)
            p5 = cv2.addWeighted(frame,1.0, trail_rgb,0.85,0); put_kv(p5,10,20,"RAW ⊙ TRAIL")

            grid = tile2x3(p0,p1,p2,p3,ov,p5, cell_h=CELL_H)
            cv2.putText(grid, f"frames={saved}/{frames_target}", (10, grid.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow(GRID_WIN, grid)

            # saving one sample per iteration (when contour is valid)
            if cnt is not None and saved < frames_target:
                ts = round(time.time(),3)
                img_id = f"{int(ts*1000)}_{saved:06d}"

                # raw
                path_raw = str((p_raw / f"{img_id}.jpg").resolve())
                cv2.imwrite(path_raw, frame, [cv2.IMWRITE_JPEG_QUALITY,95])

                # mask
                path_mask = str((p_mask / f"{img_id}.png").resolve())
                cv2.imwrite(path_mask, m_final)

                # overlay
                path_ovl  = str((p_ovl / f"{img_id}.jpg").resolve())
                cv2.imwrite(path_ovl, ov, [cv2.IMWRITE_JPEG_QUALITY,95])

                # crop around bbox with small padding
                pad = int(0.08*max(W,H))
                x0 = max(0, x-pad); y0 = max(0, y-pad)
                x1 = min(W, x+bw+pad); y1 = min(H, y+bh+pad)
                crop = frame[y0:y1, x0:x1]
                path_crop = str((p_crop / f"{img_id}.jpg").resolve())
                if crop.size>0: cv2.imwrite(path_crop, crop, [cv2.IMWRITE_JPEG_QUALITY,95])
                else: path_crop = ""

                # CSV row (one line per saved image)
                csv_wr.writerow([
                    img_id, ts, label,
                    W, H, round(cx,3), round(cy,3), round(cx_n,5), round(cy_n,5),
                    int(cv2.contourArea(cnt)), round(area_norm,6), round(solidity,5), round(bw/max(bh,1),5),
                    round(circ,5), x, y, bw, bh, round(extent,5), tag, fcnt, len(prior_boxes),
                    round(theta_rad,6), round(dtheta_ema,6), round(theta_accum,6), int(rot_dir),
                    path_raw, path_crop, path_mask, path_ovl
                ])
                saved += 1

            # quit early
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break

    finally:
        try: csv_fh.close()
        except: pass
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
