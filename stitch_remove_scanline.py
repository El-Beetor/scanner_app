#!/usr/bin/env python3
"""
stitch_remove_scanline.py

Fixes a thin vertical scanner/sensor artifact (e.g. a blue line) that shows up
at a fixed pixel column on every scan from a given device. If you scan the
SAME page twice with a small horizontal shift (so the area lost to the artifact
in scan 1 is clean in scan 2), the script:

  1. Aligns the two scans precisely (ORB feature matching + homography).
  2. Refines local alignment around the artifact using dense optical flow.
  3. Corrects a fixed vertical pixel offset between the scanner's left and right
     sensor halves (a hardware constant -- calibrate once with --calibrate,
     saved to scanner_config.json and applied automatically on every future run).
  4. Matches local brightness/contrast of the patch to surrounding columns.
  5. Feathers the patch into the base image.

Usage:
    # First-time calibration (measure and save the sensor offset):
    python3 stitch_remove_scanline.py img1.jpg img2.jpg --calibrate

    # Normal use:
    python3 stitch_remove_scanline.py img1.jpg img2.jpg -o result.jpg

    Optional flags:
      --rotate-deg {0,90,180,270}   rotation to apply to img2 (default 0)
      --feather N                   blend-feather width in px (default 40)
      --vertical-offset N           override saved sensor offset in pixels
      --calibrate                   measure offset, save to scanner_config.json, exit
      --debug                       save extra debug images
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "scanner_config.json")


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"ERROR: could not read image: {path}")
    return img


def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved to {CONFIG_PATH}")


def detect_blue_line(img, min_strength=500, max_width=500):
    """
    Looks for a thin vertical blue artifact line. max_width guards against
    picking up large blue regions (e.g. scanner head visible at top of frame).
    """
    h, w = img.shape[:2]
    scan_region = img[int(h * 0.10):, :]

    b, g, r = cv2.split(scan_region.astype(np.int16))
    blueness = np.clip(b - np.maximum(r, g), 0, None)
    col_score = blueness.sum(axis=0).astype(np.float64)

    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(col_score, kernel, mode="same")

    peak_col = int(np.argmax(smoothed))
    peak_val = smoothed[peak_col]
    background = np.median(smoothed)

    if peak_val < min_strength or peak_val < background * 8:
        return None

    threshold = background + (peak_val - background) * 0.12
    x_start = peak_col
    while x_start > 0 and smoothed[x_start - 1] > threshold:
        x_start -= 1
    x_end = peak_col
    while x_end < len(smoothed) - 1 and smoothed[x_end + 1] > threshold:
        x_end += 1

    if (x_end - x_start + 1) > max_width:
        return None

    return (x_start, x_end)


def measure_vertical_offset(img, x_line_start, x_line_end, sample_width=150, sample_height=800):
    """
    Measures the vertical pixel offset between the scanner's left and right
    sensor halves by cross-correlating strips on each side of the blue line.

    Returns offset in pixels (positive = right side is shifted downward).
    """
    h, w = img.shape[:2]
    mid_y = h // 2

    lx0 = max(0, x_line_start - sample_width)
    lx1 = x_line_start
    rx0 = x_line_end + 1
    rx1 = min(w, x_line_end + 1 + sample_width)

    y0 = max(0, mid_y - sample_height // 2)
    y1 = min(h, mid_y + sample_height // 2)

    left_strip = cv2.cvtColor(img[y0:y1, lx0:lx1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    right_strip = cv2.cvtColor(img[y0:y1, rx0:rx1], cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Resize to same width if strips differ
    if left_strip.shape[1] != right_strip.shape[1]:
        w_min = min(left_strip.shape[1], right_strip.shape[1])
        left_strip = left_strip[:, :w_min]
        right_strip = right_strip[:, :w_min]

    shift, _ = cv2.phaseCorrelate(left_strip, right_strip)
    vertical_offset = round(shift[1])
    print(f"  measured vertical offset: {vertical_offset}px (right side shifted {'down' if vertical_offset > 0 else 'up'})")
    return vertical_offset


def apply_vertical_offset(img, x_split, offset):
    """
    Corrects the scanner's inter-sensor vertical offset by shifting everything
    to the right of x_split upward (if offset > 0) or downward (if offset < 0).
    The gap introduced at the edge is filled by replicating the border row.
    """
    if offset == 0:
        return img

    result = img.copy()
    right = img[:, x_split:].copy()
    h = right.shape[0]

    if offset > 0:
        # right side is too low — shift it up
        result[:h - offset, x_split:] = right[offset:, :]
        result[h - offset:, x_split:] = right[-1:, :]
    else:
        # right side is too high — shift it down
        o = -offset
        result[o:, x_split:] = right[:h - o, :]
        result[:o, x_split:] = right[:1, :]

    return result


def align_to_base(base_img, moving_img, min_matches=15):
    h, w = base_img.shape[:2]
    gray_base = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    gray_moving = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=8000)
    kp1, des1 = orb.detectAndCompute(gray_base, None)
    kp2, des2 = orb.detectAndCompute(gray_moving, None)

    if des1 is not None and des2 is not None and len(kp1) > min_matches and len(kp2) > min_matches:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)
        good = matches[: max(min_matches, int(len(matches) * 0.25))]

        if len(good) >= min_matches:
            pts_base = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_moving = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(pts_moving, pts_base, cv2.RANSAC, 5.0)

            if H is not None:
                inliers = int(mask.sum()) if mask is not None else 0
                print(f"  homography alignment: {inliers}/{len(good)} inlier matches")
                return cv2.warpPerspective(moving_img, H, (w, h))

    print("  WARNING: feature-based alignment failed, falling back to translation-only")
    shift, response = cv2.phaseCorrelate(
        gray_base.astype(np.float32), gray_moving.astype(np.float32)
    )
    print(f"  translation fallback: shift={shift}, confidence={response:.2f}")
    M = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
    return cv2.warpAffine(moving_img, M, (w, h))


def local_warp_patch(base, patch, x_start, x_end, context_width=200):
    h, w = base.shape[:2]
    rx0 = max(0, x_start - context_width)
    rx1 = min(w, x_end + 1 + context_width)

    gray_base = cv2.cvtColor(base[:, rx0:rx1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_patch = cv2.cvtColor(patch[:, rx0:rx1], cv2.COLOR_BGR2GRAY).astype(np.float32)

    flow = cv2.calcOpticalFlowFarneback(
        gray_base, gray_patch,
        None,
        pyr_scale=0.5, levels=5, winsize=33,
        iterations=10, poly_n=7, poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )

    roi_w = rx1 - rx0
    grid_x, grid_y = np.meshgrid(np.arange(roi_w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]

    warped_roi = cv2.remap(patch[:, rx0:rx1], map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    result = patch.copy()
    result[:, rx0:rx1] = warped_roi
    return result


def match_local_tone(base, patch, x_start, x_end, context_width=60):
    h, w = base.shape[:2]
    left = max(0, x_start - context_width)
    right = min(w, x_end + 1 + context_width)

    ref_cols = np.concatenate([
        base[:, left:x_start].reshape(-1, 3),
        base[:, x_end + 1:right].reshape(-1, 3)
    ], axis=0).astype(np.float32)

    patch_cols = patch[:, x_start:x_end + 1].reshape(-1, 3).astype(np.float32)

    adjusted = patch.astype(np.float32).copy()
    for c in range(3):
        ref_mean, ref_std = ref_cols[:, c].mean(), ref_cols[:, c].std() + 1e-6
        src_mean, src_std = patch_cols[:, c].mean(), patch_cols[:, c].std() + 1e-6
        scale = ref_std / src_std
        shift_val = ref_mean - src_mean * scale
        adjusted[:, x_start:x_end + 1, c] = (
            patch[:, x_start:x_end + 1, c].astype(np.float32) * scale + shift_val
        )

    return np.clip(adjusted, 0, 255).astype(np.uint8)


def feather_patch(base, patch, x_start, x_end, feather=40):
    out = base.astype(np.float32).copy()
    patch_f = patch.astype(np.float32)
    h, w = base.shape[:2]

    fx_start = max(0, x_start - feather)
    fx_end = min(w - 1, x_end + feather)

    for x in range(fx_start, fx_end + 1):
        if x < x_start:
            alpha = (x - fx_start) / max(1, (x_start - fx_start))
        elif x > x_end:
            alpha = 1 - (x - x_end) / max(1, (fx_end - x_end))
        else:
            alpha = 1.0
        out[:, x] = (1 - alpha) * out[:, x] + alpha * patch_f[:, x]

    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image1", help="first scan (base orientation)")
    ap.add_argument("image2", help="second scan (shifted slightly to cover the artifact)")
    ap.add_argument("-o", "--output", default="stitched_result.jpg")
    ap.add_argument("--rotate-deg", type=int, choices=[0, 90, 180, 270], default=0)
    ap.add_argument("--feather", type=int, default=40)
    ap.add_argument("--vertical-offset", type=int, default=None,
                    help="override saved sensor offset (pixels, positive = right side too low)")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure vertical sensor offset, save to scanner_config.json, and exit")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    img1 = load_image(args.image1)

    print("Detecting artifact line in image1...")
    line1 = detect_blue_line(img1)
    if line1 is None:
        sys.exit("ERROR: could not detect a blue artifact line in image1")
    print(f"  found at columns {line1[0]}-{line1[1]} (width {line1[1]-line1[0]+1}px)")

    if args.calibrate:
        if args.vertical_offset is not None:
            offset = args.vertical_offset
            print(f"Saving manually specified vertical offset: {offset}px")
        else:
            print("Calibrating vertical sensor offset automatically...")
            offset = measure_vertical_offset(img1, line1[0], line1[1])
            print("  NOTE: auto-measurement may be unreliable if left/right content differs.")
            print("  If the result looks wrong, re-run with --calibrate --vertical-offset N")
        cfg = load_config()
        cfg["vertical_offset"] = offset
        save_config(cfg)
        print(f"Calibration complete. vertical_offset={offset}px saved to scanner_config.json.")
        return

    img2 = load_image(args.image2)

    rot_map = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    img2_oriented = img2 if rot_map[args.rotate_deg] is None else cv2.rotate(img2, rot_map[args.rotate_deg])
    if img2_oriented.shape[:2] != img1.shape[:2]:
        img2_oriented = cv2.resize(img2_oriented, (img1.shape[1], img1.shape[0]))

    # Determine vertical offset to use
    if args.vertical_offset is not None:
        v_offset = args.vertical_offset
        print(f"Using vertical offset from --vertical-offset flag: {v_offset}px")
    else:
        cfg = load_config()
        if "vertical_offset" in cfg:
            v_offset = cfg["vertical_offset"]
            print(f"Using saved vertical offset: {v_offset}px")
        else:
            print("No saved vertical offset found. Run with --calibrate first, or pass --vertical-offset N.")
            v_offset = 0

    print("Correcting vertical sensor offset in image1...")
    img1_corrected = apply_vertical_offset(img1, line1[1] + 1, v_offset)

    print("Aligning image2 onto image1's frame...")
    warped2 = align_to_base(img1_corrected, img2_oriented)

    line2_in_warped = detect_blue_line(warped2)
    if line2_in_warped is not None:
        overlap = not (line2_in_warped[1] < line1[0] or line2_in_warped[0] > line1[1])
        if overlap:
            print("  WARNING: image2's artifact line overlaps image1's -- patch may be incomplete")
        else:
            print(f"  image2's own line at columns {line2_in_warped[0]}-{line2_in_warped[1]} (no overlap, good)")

    print("Applying local optical-flow warp to patch region...")
    warped2_local = local_warp_patch(img1_corrected, warped2, line1[0], line1[1])

    print("Matching local tone of patch to surrounding image...")
    warped2_toned = match_local_tone(img1_corrected, warped2_local, line1[0], line1[1])

    print("Blending with feather...")
    result = feather_patch(img1_corrected, warped2_toned, line1[0], line1[1], feather=args.feather)

    cv2.imwrite(args.output, result)
    print(f"Saved result: {args.output}")

    if args.debug:
        debug = img1_corrected.copy()
        cv2.rectangle(debug, (line1[0], 0), (line1[1], debug.shape[0] - 1), (0, 0, 255), 3)
        cv2.imwrite("debug_detected_line.jpg", debug)
        cv2.imwrite("debug_corrected_base.jpg", img1_corrected)
        cv2.imwrite("debug_warped_image2.jpg", warped2)
        cv2.imwrite("debug_local_warped.jpg", warped2_local)
        cv2.imwrite("debug_toned_patch.jpg", warped2_toned)
        print("Saved debug images.")


if __name__ == "__main__":
    main()
