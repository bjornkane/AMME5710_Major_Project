# collect_gestures_modeB_layout_static_combo_v8_trail_with_gesture.py
#
# Additions: Trackbars for tuning, basic gesture extraction (finger count via convex defects),
# overlays gesture label. Rest is same as original.

import argparse, time
import cv2
import numpy as np
from collections import deque

# ------------------ Static thresholds (initial) ------------------
HSV1_LO = np.array([  0,   0,  80], dtype=np.uint8)
HSV1_HI = np.array([ 29, 255, 255], dtype=np.uint8)
HSV2_LO = np.array([170,  15,  40], dtype=np.uint8)
HSV2_HI = np.array([179, 255, 255], dtype=np.uint8)

YCC_LO = np.array([ 10, 100,  75], dtype=np.uint8)
YCC_HI = np.array([235, 180, 135], dtype=np.uint8)

KERNEL       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

LABEL_FONT_SCALE = 0.5
LABEL_THICK      = 1

# Trackbar window
def nothing(x): pass
cv2.namedWindow('Tune HSV/YCrCb')
cv2.createTrackbar('H1L', 'Tune HSV/YCrCb', HSV1_LO[0], 180, nothing)
cv2.createTrackbar('S1L', 'Tune HSV/YCrCb', HSV1_LO[1], 255, nothing)
cv2.createTrackbar('V1L', 'Tune HSV/YCrCb', HSV1_LO[2], 255, nothing)
cv2.createTrackbar('H1H', 'Tune HSV/YCrCb', HSV1_HI[0], 180, nothing)
cv2.createTrackbar('S1H', 'Tune HSV/YCrCb', HSV1_HI[1], 255, nothing)
cv2.createTrackbar('V1H', 'Tune HSV/YCrCb', HSV1_HI[2], 255, nothing)
cv2.createTrackbar('H2L', 'Tune HSV/YCrCb', HSV2_LO[0], 180, nothing)
cv2.createTrackbar('S2L', 'Tune HSV/YCrCb', HSV2_LO[1], 255, nothing)
cv2.createTrackbar('V2L', 'Tune HSV/YCrCb', HSV2_LO[2], 255, nothing)
cv2.createTrackbar('H2H', 'Tune HSV/YCrCb', HSV2_HI[0], 180, nothing)
cv2.createTrackbar('S2H', 'Tune HSV/YCrCb', HSV2_HI[1], 255, nothing)
cv2.createTrackbar('V2H', 'Tune HSV/YCrCb', HSV2_HI[2], 255, nothing)
cv2.createTrackbar('YL',  'Tune HSV/YCrCb', YCC_LO[0], 255, nothing)
cv2.createTrackbar('CrL', 'Tune HSV/YCrCb', YCC_LO[1], 255, nothing)
cv2.createTrackbar('CbL', 'Tune HSV/YCrCb', YCC_LO[2], 255, nothing)
cv2.createTrackbar('YH',  'Tune HSV/YCrCb', YCC_HI[0], 255, nothing)
cv2.createTrackbar('CrH', 'Tune HSV/YCrCb', YCC_HI[1], 255, nothing)
cv2.createTrackbar('CbH', 'Tune HSV/YCrCb', YCC_HI[2], 255, nothing)

# ------------------ UI helpers ------------------ (same as original)
def put_kv(img, x, y, text):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                LABEL_FONT_SCALE, (40, 255, 40), LABEL_THICK, cv2.LINE_AA)

def tile2x3_fixed(a, b, c, d, e, f, cell_h=200, pad=6, bg=(20,20,20)):
    # (unchanged)
    panels = [a,b,c,d,e,f]
    scaled = []
    for p in panels:
        h, w = p.shape[:2]
        if h != cell_h:
            w2 = int(round(w * (cell_h/float(h))))
            p = cv2.resize(p, (w2, cell_h), interpolation=cv2.INTER_NEAREST)
        scaled.append(p)
    colw = [max(scaled[i].shape[1] for i in range(cidx,6,3)) for cidx in range(3)]
    rowh = [cell_h, cell_h]
    canvas_w = sum(colw) + pad*4
    canvas_h = sum(rowh) + pad*3
    canvas = np.full((canvas_h, canvas_w, 3), bg, dtype=np.uint8)
    y = pad; idx = 0
    for r in range(2):
        x = pad
        for cidx in range(3):
            p = scaled[idx]; h, w = p.shape[:2]
            y0 = y + (rowh[r]-h)//2; x0 = x + (colw[cidx]-w)//2
            canvas[y0:y0+h, x0:x0+w] = p
            x += colw[cidx] + pad; idx += 1
        y += rowh[r] + pad
    return canvas

def masked_blend(bgr, mask_u8, color=(0,255,0), alpha=0.35):
    overlay = np.full_like(bgr, color, dtype=np.uint8)
    blended = cv2.addWeighted(bgr, 1.0, overlay, alpha, 0.0)
    out = bgr.copy(); m = mask_u8 > 0; out[m] = blended[m]
    return out

# ------------------ Shared preprocessing ------------------ (same)
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

def preprocess_shared(raw_bgr):
    wb   = gray_world_wb(raw_bgr)
    adj  = adaptive_gamma_from_Y(wb)
    blur = cv2.GaussianBlur(adj, (5, 5), 0)
    return blur

# ------------------ Mask utilities ------------------ (same)
def postprocess_mask(mask_raw):
    mask = (mask_raw > 0).astype(np.uint8) * 255
    mask = cv2.erode(mask, KERNEL, iterations=1)
    mask = cv2.dilate(mask, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE, iterations=1)
    return mask

def apply_right_roi(mask, frac=0.45):
    if frac <= 0.0: return mask
    h, w = mask.shape[:2]
    x0 = int(w * frac)
    roi_mask = np.zeros_like(mask); roi_mask[:, x0:] = 255
    return cv2.bitwise_and(mask, roi_mask)

def apply_vertical_gate(mask, y_lo_frac=0.0, y_hi_frac=1.0):
    h, w = mask.shape[:2]
    y0 = int(max(0.0, min(1.0, y_lo_frac)) * h)
    y1 = int(max(0.0, min(1.0, y_hi_frac)) * h)
    vmask = np.zeros_like(mask)
    vmask[y0:y1, :] = 255
    return cv2.bitwise_and(mask, vmask)

# ------------------ Mask builder (HSV + YCrCb only) ------------------
def build_masks_shared(raw_bgr, roi_x_frac=0.45, y_lo_frac=0.30, y_hi_frac=1.0):
    blur = preprocess_shared(raw_bgr)

    # Get dynamic thresholds from trackbars
    h1l = cv2.getTrackbarPos('H1L', 'Tune HSV/YCrCb')
    s1l = cv2.getTrackbarPos('S1L', 'Tune HSV/YCrCb')
    v1l = cv2.getTrackbarPos('V1L', 'Tune HSV/YCrCb')
    h1h = cv2.getTrackbarPos('H1H', 'Tune HSV/YCrCb')
    s1h = cv2.getTrackbarPos('S1H', 'Tune HSV/YCrCb')
    v1h = cv2.getTrackbarPos('V1H', 'Tune HSV/YCrCb')
    h2l = cv2.getTrackbarPos('H2L', 'Tune HSV/YCrCb')
    s2l = cv2.getTrackbarPos('S2L', 'Tune HSV/YCrCb')
    v2l = cv2.getTrackbarPos('V2L', 'Tune HSV/YCrCb')
    h2h = cv2.getTrackbarPos('H2H', 'Tune HSV/YCrCb')
    s2h = cv2.getTrackbarPos('S2H', 'Tune HSV/YCrCb')
    v2h = cv2.getTrackbarPos('V2H', 'Tune HSV/YCrCb')
    yl  = cv2.getTrackbarPos('YL',  'Tune HSV/YCrCb')
    crl = cv2.getTrackbarPos('CrL', 'Tune HSV/YCrCb')
    cbl = cv2.getTrackbarPos('CbL', 'Tune HSV/YCrCb')
    yh  = cv2.getTrackbarPos('YH',  'Tune HSV/YCrCb')
    crh = cv2.getTrackbarPos('CrH', 'Tune HSV/YCrCb')
    cbh = cv2.getTrackbarPos('CbH', 'Tune HSV/YCrCb')

    hsv1_lo = np.array([h1l, s1l, v1l])
    hsv1_hi = np.array([h1h, s1h, v1h])
    hsv2_lo = np.array([h2l, s2l, v2l])
    hsv2_hi = np.array([h2h, s2h, v2h])
    ycc_lo  = np.array([yl, crl, cbl])
    ycc_hi  = np.array([yh, crh, cbh])

    # HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, hsv1_lo, hsv1_hi)
    m2  = cv2.inRange(hsv, hsv2_lo, hsv2_hi)
    mask_hsv = postprocess_mask(cv2.bitwise_or(m1, m2))

    # YCrCb
    ycc      = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
    mask_ycc = postprocess_mask(cv2.inRange(ycc, ycc_lo, ycc_hi))

    # spatial gates
    if roi_x_frac > 0.0:
        mask_hsv = apply_right_roi(mask_hsv, roi_x_frac)
        mask_ycc = apply_right_roi(mask_ycc, roi_x_frac)
    mask_hsv = apply_vertical_gate(mask_hsv, y_lo_frac, y_hi_frac)
    mask_ycc = apply_vertical_gate(mask_ycc, y_lo_frac, y_hi_frac)

    return mask_hsv, mask_ycc

# ------------------ Fusion + scoring ------------------ (same)
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

    s_hsv, c_hsv, a_hsv = mask_score(hsv_pp, frame_shape)
    s_ycc, c_ycc, a_ycc = mask_score(ycc_pp, frame_shape)
    s_and, c_and, a_and = mask_score(and_mask, frame_shape)

    iou_hy = iou(hsv_pp, ycc_pp)

    AND_MIN     = 0.0005
    SINGLE_MIN  = 0.0007
    IOU_FOR_AND = 0.35
    SINGLE_GAP  = 1.15

    if iou_hy >= IOU_FOR_AND and s_and >= AND_MIN:
        return and_mask, "AND"
    if s_hsv >= SINGLE_MIN or s_ycc >= SINGLE_MIN:
        if s_hsv >= SINGLE_GAP * s_ycc: return hsv_pp, "HSV"
        if s_ycc >= SINGLE_GAP * s_hsv: return ycc_pp, "YCrCb"
        return (hsv_pp, "HSV") if a_hsv >= a_ycc else (ycc_pp, "YCrCb")
    return or_mask, "OR"

# ------------------ Hand-like contour selector ------------------ (tweaked min_area)
def select_hand_like(mask_bin, frame_shape):
    h, w = frame_shape[:2]
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None

    best = None
    best_score = 0.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 2000:  # Increased from 1200 for less noise
            continue
        area_ratio = area / float(h*w)
        if area_ratio > 0.18:
            continue

        hull = cv2.convexHull(c)
        hull_area = max(cv2.contourArea(hull), 1e-6)
        solidity = area / hull_area

        peri = max(cv2.arcLength(c, True), 1e-6)
        circ = (4.0*np.pi*area)/(peri*peri)
        if circ > 0.88:
            continue

        x,y,bw,bh = cv2.boundingRect(c)
        ar = bw/float(bh) if bh>0 else 0.0
        extent = area / float(bw*bh) if bw*bh>0 else 0.0

        score = (area_ratio**0.9) * (solidity**0.6) * ((1.0 - circ)**1.2) * (extent**0.4)

        if score > best_score:
            best_score = score
            best = c

    return best

# ------------------ Gesture Extraction ------------------
def extract_gesture(cnt):
    if cnt is None: return "No Hand"
    hull = cv2.convexHull(cnt, returnPoints=False)
    if len(hull) < 3: return "Unknown"
    defects = cv2.convexityDefects(cnt, hull)
    if defects is None: return "Fist"
    
    finger_count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        # Filter shallow defects (noise)
        a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c))  # cosine theorem
        if angle <= np.pi/2 and d > 2000:  # Depth threshold
            finger_count += 1
    
    if finger_count == 0: return "Fist"
    elif finger_count == 1: return "Point"
    elif finger_count == 2: return "Peace"
    elif finger_count == 3: return "Three"
    elif finger_count == 4: return "Four"
    elif finger_count >= 5: return "Open Hand"
    return "Unknown"

# ------------------ Overlays ------------------ (added gesture label)
def draw_overlays(frame_bgr, cnt, fuse_tag, gesture):
    out = frame_bgr.copy()
    put_kv(out, 10, 20, f"FEED + contour  fuse:{fuse_tag}")
    put_kv(out, 10, 50, f"Gesture: {gesture}")
    if cnt is not None:
        cv2.drawContours(out, [cnt], -1, (0, 255, 255), 2)
        M = cv2.moments(cnt)
        if abs(M["m00"]) > 1e-6:
            cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
            cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
    return out

def colorize_trail(trail_canvas):
    colored = np.zeros((trail_canvas.shape[0], trail_canvas.shape[1], 3), dtype=np.uint8)
    colored[:,:,1] = np.clip(trail_canvas, 0, 255)
    return colored

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--roi_frac", type=float, default=0.45)
    ap.add_argument("--y_lo", type=float, default=0.30)
    ap.add_argument("--y_hi", type=float, default=1.00)
    ap.add_argument("--cell_h", type=int, default=200)
    ap.add_argument("--trail_decay", type=float, default=0.92)
    ap.add_argument("--trail_thick", type=int, default=3)
    ap.add_argument("--trail_min_speed", type=float, default=0.002)
    ap.add_argument("--trail_idle_frames", type=int, default=25)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        print("ERROR: camera not available. Try --cam 1 or close other apps.")
        return

    trail_pts = deque(maxlen=20)
    trail_canvas = None
    idle_counter = 0

    try:
        while True:
            ok, frame_raw = cap.read()
            if not ok: break

            raw = cv2.flip(frame_raw, 1)
            h, w = raw.shape[:2]
            if trail_canvas is None:
                trail_canvas = np.zeros((h, w), dtype=np.uint8)

            mask_hsv_pp, mask_ycc_pp = build_masks_shared(raw, args.roi_frac, args.y_lo, args.y_hi)

            mask_fused, fuse_tag = auto_fuse_hsv_ycc(mask_hsv_pp, mask_ycc_pp, raw.shape)

            cnt_keep = select_hand_like(mask_fused, raw.shape)

            # Extract gesture
            gesture = extract_gesture(cnt_keep)

            mask_final = np.zeros_like(mask_fused)
            if cnt_keep is not None:
                cv2.drawContours(mask_final, [cnt_keep], -1, 255, -1)

            # Trail (same, but with motion check)
            if cnt_keep is not None:
                M = cv2.moments(cnt_keep)
                if abs(M["m00"]) > 1e-6:
                    cx = M["m10"]/M["m00"]; cy = M["m01"]/M["m00"]
                    trail_pts.append((cx, cy))
                cv2.multiply(trail_canvas, np.array([args.trail_decay], dtype=np.float32), dst=trail_canvas)
                if len(trail_pts) >= 2:
                    x0,y0 = trail_pts[-2]; x1,y1 = trail_pts[-1]
                    dist = np.hypot(x1-x0, y1-y0) / np.hypot(w, h)
                    if dist >= args.trail_min_speed:
                        cv2.line(trail_canvas, (int(x0),int(y0)), (int(x1),int(y1)), color=255, thickness=args.trail_thick)
                        idle_counter = 0
                    else:
                        idle_counter += 1
                else:
                    idle_counter += 1
            else:
                cv2.multiply(trail_canvas, np.array([args.trail_decay], dtype=np.float32), dst=trail_canvas)
                idle_counter += 1

            if idle_counter >= args.trail_idle_frames:
                trail_canvas[:] = 0
                idle_counter = 0

            # Panels (updated feed_overlay with gesture)
            raw_panel   = raw.copy(); put_kv(raw_panel, 10, 20, "RAW")
            hsv_panel   = cv2.cvtColor(mask_hsv_pp, cv2.COLOR_GRAY2BGR); put_kv(hsv_panel, 10, 20, "HSV mask")
            ycc_panel   = cv2.cvtColor(mask_ycc_pp, cv2.COLOR_GRAY2BGR); put_kv(ycc_panel, 10, 20, "YCrCb mask")
            final_panel = cv2.cvtColor(mask_final,  cv2.COLOR_GRAY2BGR); put_kv(final_panel, 10, 20, f"FINAL (hand-only)")

            feed_overlay = draw_overlays(raw, cnt_keep, fuse_tag, gesture)

            trail_colored = colorize_trail(trail_canvas)
            raw_trail = cv2.addWeighted(raw, 1.0, trail_colored, 0.85, 0.0)
            put_kv(raw_trail, 10, 20, "RAW ⊙ TRAIL")

            grid = tile2x3_fixed(raw_panel, hsv_panel, ycc_panel,
                                 final_panel, feed_overlay, raw_trail,
                                 cell_h=args.cell_h)
            cv2.imshow("layout: [RAW | HSV | YCrCb] / [FINAL | FEED+overlays | RAW⊙TRAIL]", grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                trail_canvas[:] = 0
                trail_pts.clear()
                idle_counter = 0

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()