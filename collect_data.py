# collect_data.py
# MediaPipe-only collector with HSV∨YCrCb masking.
# Backward-compatible: accepts --prior but only "mediapipe" is allowed and ignored.

import argparse, time, os, csv
import cv2
import numpy as np
from collections import deque
from typing import List, Tuple

HSV1_LO = np.array([  0,   12,  71], dtype=np.uint8)
HSV1_HI = np.array([ 29, 255, 255], dtype=np.uint8)
HSV2_LO = np.array([170,  15,  54], dtype=np.uint8)
HSV2_HI = np.array([179, 255, 255], dtype=np.uint8)
YCC_LO  = np.array([ 10, 128, 121], dtype=np.uint8)
YCC_HI  = np.array([ 69, 255, 191], dtype=np.uint8)

KERNEL       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
LABEL_FONT_SCALE = 0.5
LABEL_THICK      = 1

TUNE_WIN = "Tune HSV/YCrCb"
GRID_WIN = "layout: [RAW | HSV | YCrCb] / [FINAL | FEED+PRIOR | RAW⊙TRAIL]"

def init_tuner(win_name=TUNE_WIN, w=420, h=480, x=20, y=40):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, w, h)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, np.full((h, w, 3), 30, np.uint8))
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

def ensure_grid_window(x=460, y=40):
    cv2.namedWindow(GRID_WIN, cv2.WINDOW_NORMAL)
    cv2.moveWindow(GRID_WIN, x, y)

def nothing(x): pass

def build_trackbars():
    cv2.createTrackbar('H1L', TUNE_WIN, int(HSV1_LO[0]), 180, nothing)
    cv2.createTrackbar('S1L', TUNE_WIN, int(HSV1_LO[1]), 255, nothing)
    cv2.createTrackbar('V1L', TUNE_WIN, int(HSV1_LO[2]), 255, nothing)
    cv2.createTrackbar('H1H', TUNE_WIN, int(HSV1_HI[0]), 180, nothing)
    cv2.createTrackbar('S1H', TUNE_WIN, int(HSV1_HI[1]), 255, nothing)
    cv2.createTrackbar('V1H', TUNE_WIN, int(HSV1_HI[2]), 255, nothing)
    cv2.createTrackbar('H2L', TUNE_WIN, int(HSV2_LO[0]), 180, nothing)
    cv2.createTrackbar('S2L', TUNE_WIN, int(HSV2_LO[1]), 255, nothing)
    cv2.createTrackbar('V2L', TUNE_WIN, int(HSV2_LO[2]), 255, nothing)
    cv2.createTrackbar('H2H', TUNE_WIN, int(HSV2_HI[0]), 180, nothing)
    cv2.createTrackbar('S2H', TUNE_WIN, int(HSV2_HI[1]), 255, nothing)
    cv2.createTrackbar('V2H', TUNE_WIN, int(HSV2_HI[2]), 255, nothing)
    cv2.createTrackbar('YL',  TUNE_WIN, int(YCC_LO[0]), 255, nothing)
    cv2.createTrackbar('CrL', TUNE_WIN, int(YCC_LO[1]), 255, nothing)
    cv2.createTrackbar('CbL', TUNE_WIN, int(YCC_LO[2]), 255, nothing)
    cv2.createTrackbar('YH',  TUNE_WIN, int(YCC_HI[0]), 255, nothing)
    cv2.createTrackbar('CrH', TUNE_WIN, int(YCC_HI[1]), 255, nothing)
    cv2.createTrackbar('CbH', TUNE_WIN, int(YCC_HI[2]), 255, nothing)

def put_kv(img, x, y, text):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE, (40, 255, 40), LABEL_THICK, cv2.LINE_AA)

def tile2x3_fixed(a, b, c, d, e, f, cell_h=100, pad=6, bg=(20,20,20)):
    panels = [a,b,c,d,e,f]
    scaled = []
    for p in panels:
        h, w = p.shape[:2]
        if h != cell_h:
            w2 = int(round(w * (cell_h/float(h))))
            p = cv2.resize(p, (w2, cell_h), interpolation=cv2.INTER_NEAREST)
        scaled.append(p)
    colw = [max(s.shape[1] for s in scaled[i::3]) for i in range(3)]
    canvas_w = sum(colw) + pad*4
    canvas_h = cell_h*2 + pad*3
    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    y = pad; idx = 0
    for _ in range(2):
        x = pad
        for cidx in range(3):
            p = scaled[idx]; h, w = p.shape[:2]
            y0 = y + (cell_h-h)//2; x0 = x + (colw[cidx]-w)//2
            canvas[y0:y0+h, x0:x0+w] = p
            x += colw[cidx] + pad; idx += 1
        y += cell_h + pad
    return canvas

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
    lut = np.array([(i/255.0)**inv * 255.0 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(bgr, lut)

def postprocess_mask(mask_raw):
    mask = (mask_raw > 0).astype(np.uint8) * 255
    mask = cv2.erode(mask, KERNEL, iterations=1)
    mask = cv2.dilate(mask, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE, iterations=1)
    return mask

def apply_vertical_gate(mask, y_lo_frac=0.0, y_hi_frac=1.0):
    h, w = mask.shape[:2]
    y0 = int(max(0.0, min(1.0, y_lo_frac)) * h)
    y1 = int(max(0.0, min(1.0, y_hi_frac)) * h)
    vmask = np.zeros_like(mask)
    vmask[y0:y1, :] = 255
    return cv2.bitwise_and(mask, vmask)

def read_trackbar_thresholds():
    get = cv2.getTrackbarPos
    h1l,s1l,v1l = get('H1L',TUNE_WIN),get('S1L',TUNE_WIN),get('V1L',TUNE_WIN)
    h1h,s1h,v1h = get('H1H',TUNE_WIN),get('S1H',TUNE_WIN),get('V1H',TUNE_WIN)
    h2l,s2l,v2l = get('H2L',TUNE_WIN),get('S2L',TUNE_WIN),get('V2L',TUNE_WIN)
    h2h,s2h,v2h = get('H2H',TUNE_WIN),get('S2H',TUNE_WIN),get('V2H',TUNE_WIN)
    yl,crl,cbl  = get('YL',TUNE_WIN),get('CrL',TUNE_WIN),get('CbL',TUNE_WIN)
    yh,crh,cbh  = get('YH',TUNE_WIN),get('CrH',TUNE_WIN),get('CbH',TUNE_WIN)
    return (np.array([h1l,s1l,v1l]), np.array([h1h,s1h,v1h]),
            np.array([h2l,s2l,v2l]), np.array([h2h,s2h,v2h]),
            np.array([yl,crl,cbl]),  np.array([yh,crh,cbh]))

def build_masks_shared(raw_bgr, y_lo_frac=0.30, y_hi_frac=1.0):
    wb   = gray_world_wb(raw_bgr)
    adj  = adaptive_gamma_from_Y(wb)
    blur = cv2.GaussianBlur(adj, (5, 5), 0)
    hsv1_lo,hsv1_hi,hsv2_lo,hsv2_hi,ycc_lo,ycc_hi = read_trackbar_thresholds()
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, hsv1_lo, hsv1_hi)
    m2  = cv2.inRange(hsv, hsv2_lo, hsv2_hi)
    mask_hsv = postprocess_mask(cv2.bitwise_or(m1, m2))
    ycc      = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    mask_ycc = postprocess_mask(cv2.inRange(ycc, ycc_lo, ycc_hi))
    mask_hsv = apply_vertical_gate(mask_hsv, y_lo_frac, y_hi_frac)
    mask_ycc = apply_vertical_gate(mask_ycc, y_lo_frac, y_hi_frac)
    return mask_hsv, mask_ycc

def largest_blob(mask_bin, min_area=2000, max_area_abs=np.inf):
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best = None; best_a = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a >= min_area and a <= max_area_abs and a > best_a:
            best = c; best_a = a
    return best

def mask_score(mask_bin, frame_shape):
    h, w = frame_shape[:2]
    frame_area = float(h*w)
    cnt = largest_blob(mask_bin, min_area=1200, max_area_abs=0.6*frame_area)
    if cnt is None: return 0.0, None, 0.0
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = max(cv2.contourArea(hull), 1e-6)
    solidity = area / hull_area
    peri = max(cv2.arcLength(cnt, True), 1e-6)
    circularity = (4.0*np.pi*area)/(peri*peri)
    circularity = float(np.clip(circularity, 0.2, 1.0))
    area_ratio = area / frame_area
    score = float(area_ratio * solidity * circularity)
    return score, cnt, area_ratio

def iou(a, b):
    inter = cv2.bitwise_and(a, b)
    union = cv2.bitwise_or(a, b)
    u = max(int(np.count_nonzero(union)), 1)
    return float(np.count_nonzero(inter)) / float(u)

def auto_fuse_hsv_ycc(hsv_pp, ycc_pp, frame_shape):
    and_mask = cv2.bitwise_and(hsv_pp, ycc_pp)
    or_mask  = cv2.bitwise_or(hsv_pp,  ycc_pp)
    s_hsv, _, _ = mask_score(hsv_pp, frame_shape)
    s_ycc, _, _ = mask_score(ycc_pp, frame_shape)
    s_and, _, _ = mask_score(and_mask, frame_shape)
    iou_hy = iou(hsv_pp, ycc_pp)
    AND_MIN, SINGLE_MIN, IOU_FOR_AND, SINGLE_GAP = 0.0005, 0.0007, 0.35, 1.15
    if iou_hy >= IOU_FOR_AND and s_and >= AND_MIN: return and_mask, "AND"
    if s_hsv >= SINGLE_MIN or s_ycc >= SINGLE_MIN:
        if s_hsv >= SINGLE_GAP * s_ycc: return hsv_pp, "HSV"
        if s_ycc >= SINGLE_GAP * s_hsv: return ycc_pp, "YCrCb"
        return (hsv_pp, "HSV")
    return or_mask, "OR"

def select_hand_like(mask_bin, frame_shape):
    h, w = frame_shape[:2]
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    best = None; best_score = 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2000: continue
        area_ratio = area / float(h*w)
        if area_ratio > 0.18: continue
        hull = cv2.convexHull(c)
        hull_area = max(cv2.contourArea(hull), 1e-6)
        solidity = area / hull_area
        peri = max(cv2.arcLength(c, True), 1e-6)
        circ = (4.0*np.pi*area)/(peri*peri)
        if circ > 0.88: continue
        x,y,bw,bh = cv2.boundingRect(c)
        extent = area / float(bw*bh) if bw*bh>0 else 0.0
        score = (area_ratio**0.9) * (solidity**0.6) * ((1.0 - circ)**1.2) * (extent**0.4)
        if score > best_score: best_score, best = score, c
    return best

def safe_convexity_defects(cnt):
    if cnt is None or len(cnt) < 3:
        return None, None
    cnt_i32 = cnt.astype(np.int32)
    def try_defects(c):
        if c is None or len(c) < 3: return None
        h = cv2.convexHull(c, returnPoints=False)
        if h is None or len(h) < 3: return None
        h = h.reshape(-1).astype(np.int32)
        h = np.unique(h)
        if len(h) < 3: return None
        h = np.sort(h).reshape(-1, 1)
        try:
            return cv2.convexityDefects(c, h), c
        except cv2.error:
            return None
    out = try_defects(cnt_i32)
    if out is not None:
        defects, contour_used = out
        return defects, contour_used
    approx = cv2.approxPolyDP(cnt_i32, 2.0, True)
    out = try_defects(approx)
    if out is not None:
        defects, contour_used = out
        return defects, contour_used
    return None, None

def extract_gesture(cnt):
    if cnt is None: return "No Hand"
    defects, used = safe_convexity_defects(cnt)
    if defects is None:
        area = cv2.contourArea(cnt)
        peri  = max(cv2.arcLength(cnt, True), 1e-6)
        circ  = (4.0*np.pi*area)/(peri*peri)
        return "Fist" if circ > 0.85 else "Unknown"
    c = used if used is not None else cnt
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        a = np.linalg.norm(np.array(c[e][0]) - np.array(c[s][0]))
        b = np.linalg.norm(np.array(c[f][0]) - np.array(c[s][0]))
        cseg = np.linalg.norm(np.array(c[e][0]) - np.array(c[f][0]))
        if b <= 1e-6 or cseg <= 1e-6: continue
        angle = np.arccos(np.clip((b*b + cseg*cseg - a*a) / (2*b*cseg), -1.0, 1.0))
        if angle <= np.pi/2 and d > 2000:
            finger_count += 1
    if finger_count == 0: return "Fist"
    if finger_count == 1: return "Point"
    if finger_count == 2: return "Peace"
    if finger_count == 3: return "Three"
    if finger_count == 4: return "Four"
    if finger_count >= 5: return "Open Hand"
    return "Unknown"

class MPPalmDetector:
    def __init__(self, max_hands=2, det_conf=0.5, track_conf=0.5):
        self.ok = False
        self.last_polys: List[np.ndarray] = []
        try:
            import mediapipe as mp
            self.mp = mp
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                min_detection_confidence=det_conf,
                min_tracking_confidence=track_conf
            )
            self.ok = True
        except Exception:
            self.ok = False
            self.hands = None
            self.mp = None

    def infer(self, bgr: np.ndarray):
        self.last_polys = []
        if not self.ok:
            return []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)
        out = []
        if res and res.multi_hand_landmarks and res.multi_handedness:
            h, w = bgr.shape[:2]
            for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                xs = [int(p.x * w) for p in lm.landmark]
                ys = [int(p.y * h) for p in lm.landmark]
                x1, y1 = max(0,min(xs)), max(0,min(ys))
                x2, y2 = min(w,max(xs)), min(h,max(ys))
                conf = handed.classification[0].score if handed and handed.classification else 0.9
                out.append((x1,y1,x2,y2,"hand", float(conf)))
                pts = np.array([[xs[i], ys[i]] for i in range(len(xs))], dtype=np.int32)
                hull = cv2.convexHull(pts)
                self.last_polys.append(hull.reshape(-1,1,2))
        return out

def gate_mask_with_polys(mask: np.ndarray, polys: List[np.ndarray], inflate_px: int=6) -> np.ndarray:
    if not polys:
        return mask
    h, w = mask.shape[:2]
    gate = np.zeros((h,w), dtype=np.uint8)
    for poly in polys:
        if poly is None or len(poly) < 3:
            continue
        cv2.fillPoly(gate, [poly], 255)
    if inflate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inflate_px*2+1, inflate_px*2+1))
        gate = cv2.dilate(gate, kernel, iterations=1)
    return cv2.bitwise_and(mask, gate)

def draw_boxes(img: np.ndarray, dets, color=(255,200,0)) -> np.ndarray:
    out = img.copy()
    for (x1,y1,x2,y2,label,score) in dets:
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"{label}:{score:.2f}", (x1, max(15,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return out

def draw_overlays(frame_bgr, cnt, fuse_tag, gesture, dets):
    out = frame_bgr.copy()
    put_kv(out, 10, 20, f"FEED + contour  fuse:{fuse_tag}")
    put_kv(out, 10, 50, f"Gesture: {gesture}")
    out = draw_boxes(out, dets, color=(255,200,0))
    if cnt is not None:
        cv2.drawContours(out, [cnt], -1, (0, 255, 255), 2)
        M = cv2.moments(cnt)
        if abs(M["m00"]) > 1e-6:
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
    return out

def ensure_paths(save_dir: str):
    img_raw = os.path.join(save_dir, "images_raw");   os.makedirs(img_raw, exist_ok=True)
    img_crop = os.path.join(save_dir, "images_crop"); os.makedirs(img_crop, exist_ok=True)
    img_mask = os.path.join(save_dir, "masks");       os.makedirs(img_mask, exist_ok=True)
    img_ovly = os.path.join(save_dir, "overlays");    os.makedirs(img_ovly, exist_ok=True)
    return img_raw, img_crop, img_mask, img_ovly

def ensure_csv(path, header):
    make_header = not os.path.exists(path)
    fh = open(path, "a", newline="")
    wr = csv.writer(fh)
    if make_header:
        wr.writerow(header)
    return fh, wr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--out", type=str, default="gestures_all.csv")
    ap.add_argument("--save_dir", type=str, default="dataset")
    ap.add_argument("--frames", type=int, default=-1)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--y_lo", type=float, default=0.30)
    ap.add_argument("--y_hi", type=float, default=1.00)
    ap.add_argument("--cell_h", type=int, default=200)
    ap.add_argument("--trail_decay", type=float, default=0.92)
    ap.add_argument("--trail_thick", type=int, default=3)
    ap.add_argument("--trail_min_speed", type=float, default=0.002)
    ap.add_argument("--tune_w", type=int, default=420)
    ap.add_argument("--tune_h", type=int, default=480)
    ap.add_argument("--tune_x", type=int, default=20)
    ap.add_argument("--tune_y", type=int, default=40)
    ap.add_argument("--grid_x", type=int, default=460)
    ap.add_argument("--grid_y", type=int, default=40)
    # backward-compat: accept --prior but only mediapipe
    ap.add_argument("--prior", type=str, default="mediapipe", choices=["mediapipe"],
                    help="kept for compatibility; always mediapipe")

    ap.add_argument("--mp_max_hands", type=int, default=2)
    ap.add_argument("--mp_det_conf", type=float, default=0.5)
    ap.add_argument("--mp_track_conf", type=float, default=0.5)

    ap.add_argument("--save_raw", type=int, default=1)
    ap.add_argument("--save_crop", type=int, default=1)
    ap.add_argument("--save_mask", type=int, default=1)
    ap.add_argument("--save_overlay", type=int, default=1)

    args = ap.parse_args()

    header = [
        "image_id","time_s","w","h",
        "cx","cy","cx_n","cy_n",
        "area","area_norm","solidity","aspect_ratio","circularity",
        "bbox_x","bbox_y","bbox_w","bbox_h","extent",
        "mask_fuse","finger_count",
        "prior","prior_nboxes","gesture","label",
        "path_raw","path_crop","path_mask","path_overlay"
    ]
    csv_fh, csv_wr = ensure_csv(args.out, header)
    p_raw, p_crop, p_mask, p_ovly = ensure_paths(args.save_dir)

    init_tuner(TUNE_WIN, args.tune_w, args.tune_h, args.tune_x, args.tune_y)
    build_trackbars()
    ensure_grid_window(args.grid_x, args.grid_y)

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print("ERROR: camera not available. Try --cam 1 or close other apps.")
        return

    mpdet = MPPalmDetector(max_hands=args.mp_max_hands, det_conf=args.mp_det_conf, track_conf=args.mp_track_conf)
    if not mpdet.ok:
        print("ERROR: mediapipe unavailable. pip install mediapipe")
        return

    trail_pts = deque(maxlen=20)
    trail_canvas = None
    saved_count = 0
    prior_name = "mediapipe"

    try:
        while True:
            ok, frame_raw = cap.read()
            if not ok: break
            raw = cv2.flip(frame_raw, 1)
            h, w = raw.shape[:2]
            if trail_canvas is None:
                trail_canvas = np.zeros((h, w), dtype=np.uint8)

            cv2.imshow(TUNE_WIN, np.full((args.tune_h, args.tune_w, 3), 30, np.uint8))

            mask_hsv_pp, mask_ycc_pp = build_masks_shared(raw, args.y_lo, args.y_hi)
            mask_fused, fuse_tag = auto_fuse_hsv_ycc(mask_hsv_pp, mask_ycc_pp, raw.shape)

            dets = mpdet.infer(raw)
            infl = max(6, int(0.08 * max(h, w)))
            if getattr(mpdet, "last_polys", None):
                mask_fused_gated = gate_mask_with_polys(mask_fused, mpdet.last_polys, inflate_px=infl)
            else:
                mask_fused_gated = mask_fused

            cnt_keep = select_hand_like(mask_fused_gated, raw.shape)
            gesture = extract_gesture(cnt_keep)

            if cnt_keep is not None:
                area = cv2.contourArea(cnt_keep)
                hull = cv2.convexHull(cnt_keep)
                hull_area = max(cv2.contourArea(hull), 1e-6)
                solidity = area / hull_area
                x,y,bw,bh = cv2.boundingRect(cnt_keep)
                extent = area / float(max(bw*bh, 1))
                peri = max(cv2.arcLength(cnt_keep, True), 1e-6)
                circ = (4.0*np.pi*area)/(peri*peri)
                M = cv2.moments(cnt_keep)
                if abs(M["m00"]) > 1e-6:
                    cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
                else:
                    cx, cy = -1.0, -1.0
                area_norm = area / float(w*h)
                cx_n, cy_n = (cx/float(w) if cx>=0 else -1.0), (cy/float(h) if cy>=0 else -1.0)
                aspect = bw / max(bh,1)
                finger_count_map = {"Fist":0,"Point":1,"Peace":2,"Three":3,"Four":4,"Open Hand":5}
                fcount = finger_count_map.get(gesture, -1)
                bbox_x, bbox_y, bbox_w, bbox_h = int(x), int(y), int(bw), int(bh)
            else:
                cx = cy = cx_n = cy_n = -1.0
                area = area_norm = solidity = aspect = circ = extent = -1.0
                bbox_x = bbox_y = bbox_w = bbox_h = -1
                fcount = -1
                gesture = "No Hand"

            mask_final = np.zeros_like(mask_fused_gated)
            if cnt_keep is not None:
                cv2.drawContours(mask_final, [cnt_keep], -1, 255, -1)

            if cnt_keep is not None:
                Mv = cv2.moments(cnt_keep)
                if abs(Mv["m00"]) > 1e-6:
                    cx_vis = Mv["m10"]/Mv["m00"]; cy_vis = Mv["m01"]/Mv["m00"]
                    trail_pts.append((cx_vis, cy_vis))
            cv2.multiply(trail_canvas, np.array([args.trail_decay], dtype=np.float32), dst=trail_canvas)
            if len(trail_pts) >= 2:
                x0,y0 = trail_pts[-2]; x1,y1 = trail_pts[-1]
                dist = np.hypot(x1-x0, y1-y0) / np.hypot(w, h)
                if dist >= args.trail_min_speed:
                    cv2.line(trail_canvas, (int(x0),int(y0)), (int(x1),int(y1)), color=255, thickness=args.trail_thick)

            raw_panel   = raw.copy(); put_kv(raw_panel, 10, 20, "RAW")
            hsv_panel   = cv2.cvtColor(mask_hsv_pp, cv2.COLOR_GRAY2BGR); put_kv(hsv_panel, 10, 20, "HSV mask")
            ycc_panel   = cv2.cvtColor(mask_ycc_pp, cv2.COLOR_GRAY2BGR); put_kv(ycc_panel, 10, 20, "YCrCb mask")
            final_panel = cv2.cvtColor(mask_final,  cv2.COLOR_GRAY2BGR); put_kv(final_panel, 10, 20, "FINAL (hand-only)")
            feed_overlay = draw_overlays(raw, cnt_keep, fuse_tag, gesture, dets)
            trail_colored = np.zeros_like(raw); trail_colored[:,:,1] = np.clip(trail_canvas, 0, 255)
            raw_trail = cv2.addWeighted(raw, 1.0, trail_colored, 0.85, 0.0); put_kv(raw_trail, 10, 20, "RAW ⊙ TRAIL")

            saved_count += 1
            frame_ts = round(time.time(), 3)
            img_id = f"{int(frame_ts*1000)}_{saved_count:06d}"

            path_raw = path_crop = path_mask = path_overlay = ""
            if args.save_raw:
                path_raw = os.path.join(*ensure_paths(args.save_dir)[:1], f"{img_id}.jpg")
                cv2.imwrite(path_raw, raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if args.save_mask:
                path_mask = os.path.join(ensure_paths(args.save_dir)[2], f"{img_id}.png")
                cv2.imwrite(path_mask, mask_final)
            if args.save_overlay:
                path_overlay = os.path.join(ensure_paths(args.save_dir)[3], f"{img_id}.jpg")
                cv2.imwrite(path_overlay, feed_overlay, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if args.save_crop and cnt_keep is not None:
                x,y,bw,bh = bbox_x, bbox_y, bbox_w, bbox_h
                pad = int(0.08*max(w,h))
                x0 = max(0, x - pad); y0 = max(0, y - pad)
                x1 = min(w, x + bw + pad); y1 = min(h, y + bh + pad)
                crop = raw[y0:y1, x0:x1].copy()
                path_crop = os.path.join(ensure_paths(args.save_dir)[1], f"{img_id}.jpg")
                if crop.size > 0:
                    cv2.imwrite(path_crop, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            csv_wr.writerow([
                img_id, frame_ts, w, h,
                round(cx,3), round(cy,3), round(cx_n,5), round(cy_n,5),
                int(area) if area>=0 else -1, round(area_norm,6) if area_norm>=0 else -1.0,
                round(solidity,5) if solidity>=0 else -1.0,
                round(aspect,5) if aspect>=0 else -1.0,
                round(circ,5) if circ>=0 else -1.0,
                int(bbox_x), int(bbox_y), int(bbox_w), int(bbox_h),
                round(extent,5) if extent>=0 else -1.0,
                fuse_tag,
                fcount,
                prior_name, int(len(dets)), gesture, args.label,
                path_raw, path_crop, path_mask, path_overlay
            ])

            grid = tile2x3_fixed(raw_panel, hsv_panel, ycc_panel,
                                 final_panel, feed_overlay, raw_trail,
                                 cell_h=args.cell_h)
            cv2.putText(grid, f"frames={saved_count}" + (f"/{args.frames}" if args.frames>0 else ""),
                        (10, grid.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.imshow(GRID_WIN, grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                trail_canvas[:] = 0
                trail_pts.clear()

            if args.frames > 0 and saved_count >= args.frames:
                print(f"Reached frame limit: {args.frames}")
                break

    finally:
        try: csv_fh.close()
        except Exception: pass
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
