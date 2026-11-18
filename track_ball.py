#!/usr/bin/env python3
"""Track the small black ball relative to the Van de Graaff dome, frame by frame.

For every frame the dome's silhouette and the ball's outline are each fit with a
circle at sub-pixel precision (radial edge search + robust least squares). The
ball is then expressed relative to the dome, which removes hand-held camera
shake. Camera roll is measured from the background (SIFT matched to frame 0) so
the "up" direction stays fixed even if the phone rotates.

Output columns: time, gap between the ball and dome surfaces, center-to-center
distance (dome radius + ball radius + gap), the ball's vertical acceleration,
and a flag for frames where the ball is touching the dome.

Usage:
    python3 track_ball.py IMG_5884.MOV --out ball_tracking.csv \
        [--ball-diameter-mm 14.9] [--dome-radius-m 0.1245] [--overlay-dir overlays]
"""
import argparse
import csv
import math
import os

import cv2
import numpy as np
from scipy.signal import savgol_filter

DOME_INIT = (426.1, 1577.9, 663.7)  # cx, cy, R in px for frame 0 of IMG_5884.MOV
BALL_INIT = (488.0, 860.0, 36.0)


def sample_rays(img, cx, cy, angles, radii):
    ca = np.cos(angles)[:, None]
    sa = np.sin(angles)[:, None]
    mx = (cx + radii[None, :] * ca).astype(np.float32)
    my = (cy + radii[None, :] * sa).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def subpixel_peak(score, i):
    if 0 < i < len(score) - 1:
        a, b, c = score[i - 1], score[i], score[i + 1]
        den = a - 2 * b + c
        if den < 0:
            return i + float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    return float(i)


def fit_circle(pts):
    """Algebraic (Kasa) fit followed by Gauss-Newton geometric refinement."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = -D / 2, -E / 2
    r = math.sqrt(max(cx ** 2 + cy ** 2 - F, 1e-9))
    for _ in range(10):
        dx, dy = x - cx, y - cy
        d = np.hypot(dx, dy)
        res = d - r
        J = np.column_stack([-dx / d, -dy / d, -np.ones_like(d)])
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.abs(step).max() < 1e-4:
            break
    return cx, cy, r


def robust_fit_circle(pts, min_pts):
    keep = np.ones(len(pts), bool)
    for _ in range(6):
        if keep.sum() < min_pts:
            return None
        cx, cy, r = fit_circle(pts[keep])
        res = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r
        sig = 1.4826 * np.median(np.abs(res[keep]))
        new_keep = np.abs(res) < max(3.0 * sig, 0.75)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if keep.sum() < min_pts:
        return None
    cx, cy, r = fit_circle(pts[keep])
    res = np.hypot(pts[keep, 0] - cx, pts[keep, 1] - cy) - r
    return cx, cy, r, int(keep.sum()), float(np.sqrt(np.mean(res ** 2))), keep


def fit_dome(color, prev, ball, half_window, prior_sigma):
    """Locate the dome silhouette along rays from the previous dome center."""
    h, w = color.shape[:2]
    cx, cy, R = prev
    angles = np.deg2rad(np.arange(180.0, 360.0, 0.4))
    ex = cx + R * np.cos(angles)
    ey = cy + R * np.sin(angles)
    m = 25
    ok = (ex > m) & (ex < w - m) & (ey > m) & (ey < h - m)
    if ball is not None:
        bang = math.atan2(ball[1] - cy, ball[0] - cx)
        excl = (2.2 * ball[2] + 12) / R
        dang = np.angle(np.exp(1j * (angles - bang)))
        ok &= np.abs(dang) > excl
    angles = angles[ok]
    if len(angles) < 60:
        return None, np.empty((0, 2))
    step = 0.25
    radii = R + np.arange(-half_window, half_window + step, step)
    prof = sample_rays(color, cx, cy, angles, radii)  # (n, k, 3)
    grad = np.gradient(prof, step, axis=1)
    mag = np.sqrt((grad ** 2).sum(axis=2))
    weight = np.exp(-0.5 * ((radii - R) / prior_sigma) ** 2)
    score = mag * weight[None, :]
    idx = score.argmax(axis=1)
    pts = []
    for j, i in enumerate(idx):
        ri = radii[0] + step * subpixel_peak(mag[j], i)
        pts.append((cx + ri * math.cos(angles[j]), cy + ri * math.sin(angles[j])))
    pts = np.array(pts)
    return robust_fit_circle(pts, min_pts=60), pts


def fit_ball(gray, prev, dome, half_window, prior_sigma):
    """Locate the ball outline (dark inside -> bright outside) along rays."""
    bx, by, r = prev
    angles = np.deg2rad(np.arange(0.0, 360.0, 3.0))
    step = 0.25
    radii = r + np.arange(-half_window, half_window + step, step)
    if dome is not None:
        dcx, dcy, dR = dome[:3]
        far = r + half_window
        fx = bx + far * np.cos(angles)
        fy = by + far * np.sin(angles)
        ok = np.hypot(fx - dcx, fy - dcy) > dR + 3
        angles = angles[ok]
    if len(angles) < 20:
        return None, np.empty((0, 2))
    prof = sample_rays(gray, bx, by, angles, radii)
    grad = np.gradient(prof, step, axis=1)
    weight = np.exp(-0.5 * ((radii - r) / prior_sigma) ** 2)
    score = np.clip(grad, 0, None) * weight[None, :]
    idx = score.argmax(axis=1)
    peak = grad[np.arange(len(idx)), idx]
    strong = peak > 0.35 * np.median(peak)
    pts = []
    for j, i in enumerate(idx):
        if not strong[j]:
            continue
        ri = radii[0] + step * subpixel_peak(grad[j], i)
        pts.append((bx + ri * math.cos(angles[j]), by + ri * math.sin(angles[j])))
    pts = np.array(pts)
    if len(pts) < 20:
        return None, pts
    return robust_fit_circle(pts, min_pts=20), pts


class RollTracker:
    """Camera rotation relative to frame 0 from background SIFT matches."""

    def __init__(self, scale=0.5):
        self.scale = scale
        self.sift = cv2.SIFT_create(nfeatures=2500)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ref = None

    def _features(self, gray, dome, ball):
        s = self.scale
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        mask = np.full(small.shape, 255, np.uint8)
        cv2.circle(mask, (int(dome[0] * s), int(dome[1] * s)), int((dome[2] + 30) * s), 0, -1)
        cv2.circle(mask, (int(ball[0] * s), int(ball[1] * s)), int(3 * ball[2] * s), 0, -1)
        return self.sift.detectAndCompute(small, mask)

    def update(self, gray, dome, ball):
        kp, des = self._features(gray, dome, ball)
        if self.ref is None:
            self.ref = (kp, des)
            return 0.0, 0
        rkp, rdes = self.ref
        if des is None or len(kp) < 10:
            return float("nan"), 0
        good = []
        for pair in self.matcher.knnMatch(des, rdes, k=2):
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
                good.append(pair[0])
        if len(good) < 12:
            return float("nan"), len(good)
        src = np.float32([kp[g.queryIdx].pt for g in good])
        dst = np.float32([rkp[g.trainIdx].pt for g in good])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if M is None:
            return float("nan"), 0
        return math.degrees(math.atan2(M[1, 0], M[0, 0])), int(inl.sum())


def refine(fit_fn, img, prev, other, passes, max_shift, max_dr):
    """Coarse-to-fine passes; a pass that jumps implausibly far is discarded."""
    best, best_pts, cur = None, np.empty((0, 2)), prev
    for hw, ps in passes:
        fit, pts = fit_fn(img, cur, other, hw, ps)
        if fit is None:
            break
        if math.hypot(fit[0] - prev[0], fit[1] - prev[1]) > max_shift or abs(fit[2] - prev[2]) > max_dr:
            break
        best, best_pts, cur = fit, pts, fit[:3]
    return best, best_pts


def draw_overlay(frame, dome, ball, dome_pts, ball_pts, text):
    vis = frame.copy()
    cv2.circle(vis, (round(dome[0]), round(dome[1])), round(dome[2]), (0, 255, 0), 1, cv2.LINE_AA)
    for p in dome_pts:
        cv2.circle(vis, (round(p[0]), round(p[1])), 1, (255, 0, 255), -1)
    if ball is not None:
        cv2.circle(vis, (round(ball[0]), round(ball[1])), round(ball[2]), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.drawMarker(vis, (round(ball[0]), round(ball[1])), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
    for p in ball_pts:
        cv2.circle(vis, (round(p[0]), round(p[1])), 1, (255, 255, 0), -1)
    cx = int(ball[0]) if ball is not None else 480
    cy = int(ball[1]) if ball is not None else 880
    x0, y0 = max(cx - 200, 0), max(cy - 150, 0)
    crop = vis[y0:y0 + 300, x0:x0 + 400]
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.putText(crop, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
    cv2.putText(crop, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    return crop


RAW_FIELDS = ("dome_cx", "dome_cy", "dome_r", "dome_pts", "dome_rms",
              "ball_x", "ball_y", "ball_r", "ball_pts", "ball_rms", "roll")


def track(video, max_frames=0, overlay_dir="", overlay_every=30, print_every=100):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)

    dome, ball = DOME_INIT, BALL_INIT
    roll = RollTracker()
    raw = {k: [] for k in RAW_FIELDS}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        gray8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 1.2)
        gray = cv2.GaussianBlur(gray8.astype(np.float32), (0, 0), 1.0)

        d_fit, d_pts = refine(fit_dome, color, dome, ball, ((22, 8.0), (8, 4.0), (5, 3.0)), 30, 20)
        if d_fit is not None:
            dome = d_fit[:3]
        b_fit, b_pts = refine(fit_ball, gray, ball, dome, ((14, 7.0), (8, 4.0), (6, 3.0)), 30, 5)
        if b_fit is not None:
            ball = b_fit[:3]
        roll_deg, _ = roll.update(gray8, dome, ball)

        nan5 = (np.nan,) * 5
        vals = ((d_fit[:5] if d_fit is not None else nan5) + (b_fit[:5] if b_fit is not None else nan5)
                + (roll_deg,))
        for k, v in zip(RAW_FIELDS, vals):
            raw[k].append(float(v))

        if overlay_dir and n % overlay_every == 0:
            keep_b = b_pts[b_fit[5]] if b_fit is not None else b_pts
            cv2.imwrite(os.path.join(overlay_dir, f"overlay_{n:04d}.png"),
                        draw_overlay(frame, dome, b_fit, d_pts, keep_b, f"frame {n}  t={n / fps:.2f}s"))
        if print_every and n % print_every == 0:
            print(f"frame {n}: dome center=({dome[0]:.1f}, {dome[1]:.1f}) R={dome[2]:.1f}  "
                  f"ball=({ball[0]:.2f}, {ball[1]:.2f}) r={ball[2]:.2f}")
        n += 1
    return fps, {k: np.array(v) for k, v in raw.items()}


def fill_nan(a):
    a = a.copy()
    bad = ~np.isfinite(a)
    if bad.any() and (~bad).sum() >= 2:
        idx = np.arange(len(a))
        a[bad] = np.interp(idx[bad], idx[~bad], a[~bad])
    return a


def smooth(a, window):
    window = min(window, len(a) - (1 - len(a) % 2))
    if window < 5:
        return a
    return savgol_filter(fill_nan(a), window, 2)


def derive(fps, raw, ball_diameter_mm, dome_radius_m, contact_gap_mm):
    """Per-frame quantities needed for a force-balance charge calculation."""
    n = len(raw["ball_x"])
    th = np.deg2rad(smooth(raw["roll"], 31))
    vx = raw["ball_x"] - raw["dome_cx"]
    vy = raw["ball_y"] - raw["dome_cy"]
    x = np.cos(th) * vx - np.sin(th) * vy          # right +
    y = -(np.sin(th) * vx + np.cos(th) * vy)       # up +
    mm_per_px = ball_diameter_mm / (2 * smooth(raw["ball_r"], 61))
    gap_mm = (np.hypot(x, y) - raw["dome_r"] - raw["ball_r"]) * mm_per_px
    height_m = (y - raw["dome_r"]) * mm_per_px / 1000.0
    accel = savgol_filter(fill_nan(height_m), 9, 2, deriv=2, delta=1.0 / fps)
    accel[~np.isfinite(gap_mm)] = np.nan
    return {
        "time_s": np.arange(n) / fps,
        "gap_mm": gap_mm,
        "center_distance_m": dome_radius_m + ball_diameter_mm / 2000.0 + gap_mm / 1000.0,
        "vertical_acceleration_m_per_s2": accel,
        "touching_dome": np.where(gap_mm < contact_gap_mm, 1.0, 0.0),
    }


FORMATS = {
    "time_s": ".4f",
    "gap_mm": ".3f",
    "center_distance_m": ".5f",
    "vertical_acceleration_m_per_s2": ".4f",
    "touching_dome": ".0f",
}


def write_csv(path, cols):
    names = list(cols)
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(names)
        for i in range(len(cols["time_s"])):
            wr.writerow(["" if not np.isfinite(cols[k][i]) else format(cols[k][i], FORMATS[k]) for k in names])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out", default="ball_tracking.csv")
    ap.add_argument("--ball-diameter-mm", type=float, default=14.9)
    ap.add_argument("--dome-radius-m", type=float, default=0.1245)
    ap.add_argument("--contact-gap-mm", type=float, default=0.3,
                    help="gaps below this are flagged as touching the dome")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--overlay-dir", default="", help="save fit-check images here")
    ap.add_argument("--overlay-every", type=int, default=30)
    ap.add_argument("--print-every", type=int, default=100)
    args = ap.parse_args()

    fps, raw = track(args.video, args.max_frames, args.overlay_dir, args.overlay_every, args.print_every)
    cols = derive(fps, raw, args.ball_diameter_mm, args.dome_radius_m, args.contact_gap_mm)
    write_csv(args.out, cols)
    print(f"wrote {args.out}: {len(cols['time_s'])} frames, "
          f"{int(np.isnan(cols['gap_mm']).sum())} frames without a fit")


if __name__ == "__main__":
    main()
