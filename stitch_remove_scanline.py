#!/usr/bin/env python3
"""
stitch_remove_scanline.py

Removes a fixed-column scanner sensor artifact (a thin vertical blue line) by
stitching two scans of the same page taken with a small horizontal shift, so
the columns lost to the artifact in scan 1 are clean in scan 2.

Pipeline (see run_pipeline):
  1. Correct the scanner's fixed vertical offset between its left and right
     sensor halves by shifting everything right of the artifact column.
  2. Align scan 2 onto scan 1 (ORB features + RANSAC homography).
  3. Match the patch's brightness/contrast to the surrounding columns.
  4. Feather-blend the patch over the artifact columns.

The artifact column and the vertical sensor offset are hardware constants for
a given DPI. Calibrate once per DPI (--calibrate); values are stored in
scanner_config.json next to this script and applied automatically afterwards.

Usage:
    # One-time per DPI: detect the line (use a mostly blank page)
    python3 stitch_remove_scanline.py blank.jpg --calibrate --dpi 600
    # ...then, after checking a result visually, lock in the vertical offset
    python3 stitch_remove_scanline.py blank.jpg --calibrate --dpi 600 --vertical-offset 5

    # Normal use (DPI is read from the file's JFIF header, or pass --dpi)
    python3 stitch_remove_scanline.py scan1.jpg scan2.jpg -o result.png

    Optional flags:
      --dpi N                       DPI profile to use (default: read from file)
      --rotate-deg {0,90,180,270}   rotation to apply to scan 2 (default 0)
      --feather N                   blend-feather width in px (default 40)
      --vertical-offset N           override saved sensor offset in pixels
      --calibrate                   detect line + save config for this DPI, then exit
      --debug                       save per-step debug images next to the output
"""

import argparse
import json
import os
import sys
import cv2
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scanner_config.json")


class PipelineError(RuntimeError):
    """Raised for any user-facing failure (bad input, missing calibration, ...)."""


# ── I/O and config ────────────────────────────────────────────────────────────

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise PipelineError(f"could not read image: {path}")
    return img


def read_exif_dpi(path):
    """Return DPI from the JFIF header as an int, or None if unavailable."""
    try:
        import struct
        with open(path, "rb") as f:
            data = f.read(65536)
        if data[6:10] == b"JFIF":
            units = data[13]
            x_density = struct.unpack(">H", data[14:16])[0]
            if units in (1, 2) and x_density > 0:
                return x_density if units == 1 else round(x_density * 2.54)
    except Exception:
        pass
    return None


def load_config(path=CONFIG_PATH):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_config(cfg, path=CONFIG_PATH, log=print):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    log(f"  Saved to {path}")


def dpi_key(dpi):
    return f"dpi_{dpi}"


def resolve_dpi(dpi, image_path, log=print):
    """Return the DPI profile name to use: explicit value > file header > 'default'."""
    if dpi:
        return str(dpi)
    found = read_exif_dpi(image_path)
    if found:
        log(f"  DPI from file header: {found}")
        return str(found)
    log("  WARNING: could not read DPI from file header. Using profile 'default'. "
        "Pass --dpi N if you have multiple DPI profiles.")
    return "default"


# ── Detection and calibration measurements ────────────────────────────────────

def detect_blue_line(img, min_strength=500, max_width=500):
    """
    Looks for a thin vertical blue artifact line. max_width guards against
    picking up large blue regions (e.g. scanner head visible at top of frame).
    Returns (x_start, x_end) inclusive, or None.
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


def measure_vertical_offset(img, x_line_start, x_line_end, sample_width=150,
                            sample_height=800, log=print):
    """
    Estimates the vertical pixel offset between the scanner's left and right
    sensor halves by cross-correlating strips on each side of the line.
    Positive = right side is shifted downward. Unreliable when the content on
    the two sides differs; treat as a suggestion and confirm visually.
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

    if left_strip.shape[1] != right_strip.shape[1]:
        w_min = min(left_strip.shape[1], right_strip.shape[1])
        left_strip = left_strip[:, :w_min]
        right_strip = right_strip[:, :w_min]

    shift, _ = cv2.phaseCorrelate(left_strip, right_strip)
    vertical_offset = round(shift[1])
    log(f"  measured vertical offset: {vertical_offset}px "
        f"(right side shifted {'down' if vertical_offset > 0 else 'up'})")
    return vertical_offset


# ── Pipeline stages ───────────────────────────────────────────────────────────

def apply_vertical_offset(img, x_split, offset):
    """
    Corrects the scanner's inter-sensor vertical offset by shifting everything
    to the right of x_split up (offset > 0) or down (offset < 0) by |offset| px.
    """
    if offset == 0:
        return img

    result = img.copy()
    right = img[:, x_split:].copy()
    h = right.shape[0]

    if offset > 0:
        result[:h - offset, x_split:] = right[offset:, :]
        result[h - offset:, x_split:] = right[-1:, :]
    else:
        o = -offset
        result[o:, x_split:] = right[:h - o, :]
        result[:o, x_split:] = right[:1, :]

    return result


def align_to_base(base_img, moving_img, min_matches=15, log=print):
    """Warp moving_img onto base_img's frame (ORB + RANSAC homography, with a
    phase-correlation translation fallback)."""
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
                log(f"  homography alignment: {inliers}/{len(good)} inlier matches")
                return cv2.warpPerspective(moving_img, H, (w, h))

    log("  WARNING: feature-based alignment failed, falling back to translation-only")
    shift, response = cv2.phaseCorrelate(
        gray_base.astype(np.float32), gray_moving.astype(np.float32)
    )
    log(f"  translation fallback: shift={shift}, confidence={response:.2f}")
    M = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
    return cv2.warpAffine(moving_img, M, (w, h))


def match_local_tone(base, patch, x_start, x_end, context_width=60):
    """Adjust the patch columns' mean/std per channel to match the columns of
    base immediately either side of the artifact."""
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
    """Replace columns [x_start, x_end] of base with patch, ramping alpha over
    `feather` px on each side so there is no hard seam."""
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


def save_debug_steps(out_dir, x_start, x_end, v_offset, steps, log=print):
    """Save a full annotated image and a tight seam crop for each (name, image, label)."""
    pad = 300
    for name, img, label in steps:
        h, w = img.shape[:2]
        cx0 = max(0, x_start - pad)
        cx1 = min(w, x_end + pad)

        annotated = img.copy()
        cv2.rectangle(annotated, (x_start, 0), (x_end, h - 1), (0, 0, 255), 6)
        cv2.putText(annotated, label, (50, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (0, 0, 255), 6, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out_dir, f"debug_{name}_full.jpg"), annotated)

        crop = img[:, cx0:cx1].copy()
        cv2.rectangle(crop, (x_start - cx0, 0), (x_end - cx0, crop.shape[0] - 1), (0, 0, 255), 4)
        cv2.imwrite(os.path.join(out_dir, f"debug_{name}_seam.jpg"), crop)

        log(f"  debug_{name}_full.jpg + debug_{name}_seam.jpg  —  {label}")


# ── Entry points ──────────────────────────────────────────────────────────────

def calibrate(image1, dpi=None, vertical_offset=None, config_path=CONFIG_PATH, log=print):
    """Detect the artifact line in image1 and store it (plus the vertical sensor
    offset) in the config under this DPI's profile. Returns the profile dict."""
    img1 = load_image(image1)
    cfg = load_config(config_path)
    dpi = resolve_dpi(dpi, image1, log)
    profile = cfg.get(dpi_key(dpi), {})

    if "line_start" in profile and "line_end" in profile and vertical_offset is None:
        x_start, x_end = profile["line_start"], profile["line_end"]
        log(f"Using saved line position: columns {x_start}-{x_end}")
    else:
        log("Detecting artifact line...")
        line = detect_blue_line(img1)
        if line is None:
            raise PipelineError("could not detect a blue artifact line in image1")
        x_start, x_end = line
        log(f"  found at columns {x_start}-{x_end} (width {x_end - x_start + 1}px)")

    if vertical_offset is not None:
        v_offset = vertical_offset
        log(f"Saving manually specified vertical offset: {v_offset}px")
    else:
        log("Measuring vertical sensor offset automatically...")
        v_offset = measure_vertical_offset(img1, x_start, x_end, log=log)
        log("  NOTE: auto-measurement may be unreliable if content differs across the line.")
        log("  If wrong, re-run: --calibrate --vertical-offset N")

    profile["line_start"] = x_start
    profile["line_end"] = x_end
    profile["vertical_offset"] = v_offset
    cfg[dpi_key(dpi)] = profile
    save_config(cfg, config_path, log)
    log(f"Calibration saved for DPI profile '{dpi}': "
        f"line={x_start}-{x_end}, vertical_offset={v_offset}px")
    return profile


def run_pipeline(image1, image2, output, dpi=None, vertical_offset=None, feather=40,
                 rotate_deg=0, debug=False, config_path=CONFIG_PATH, log=print):
    """
    Stitch image2's clean columns over image1's artifact line and write `output`.

    dpi:             profile to use (int/str); read from image1's header if None
    vertical_offset: override the profile's saved sensor offset (px)
    log:             callable receiving one status line at a time
    Returns a dict describing what was used. Raises PipelineError on failure.
    """
    img1 = load_image(image1)
    cfg = load_config(config_path)
    dpi = resolve_dpi(dpi, image1, log)
    profile = cfg.get(dpi_key(dpi), {})

    # Line position — always the saved calibration, never auto-detected
    if "line_start" in profile and "line_end" in profile:
        x_start, x_end = profile["line_start"], profile["line_end"]
        log(f"Using saved line position for DPI '{dpi}': columns {x_start}-{x_end} "
            f"(width {x_end - x_start + 1}px)")
    else:
        raise PipelineError(
            f"no calibration found for DPI '{dpi}'.\n"
            f"Run:  python3 stitch_remove_scanline.py scan1.jpg --calibrate --dpi {dpi}"
        )

    # Vertical offset
    if vertical_offset is not None:
        v_offset = vertical_offset
        log(f"Using vertical offset override: {v_offset}px")
    elif "vertical_offset" in profile:
        v_offset = profile["vertical_offset"]
        log(f"Using saved vertical offset for DPI '{dpi}': {v_offset}px")
    else:
        log("No saved vertical offset. Run --calibrate --vertical-offset N to set it.")
        v_offset = 0

    img2 = load_image(image2)
    rot_map = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
               180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    img2_oriented = img2 if rot_map[rotate_deg] is None else cv2.rotate(img2, rot_map[rotate_deg])
    if img2_oriented.shape[:2] != img1.shape[:2]:
        img2_oriented = cv2.resize(img2_oriented, (img1.shape[1], img1.shape[0]))

    log("Correcting vertical sensor offset in image1...")
    img1_corrected = apply_vertical_offset(img1, x_start, v_offset)

    log("Aligning image2 onto image1's frame...")
    warped2 = align_to_base(img1_corrected, img2_oriented, log=log)

    line2_in_warped = detect_blue_line(warped2)
    if line2_in_warped is not None:
        overlap = not (line2_in_warped[1] < x_start or line2_in_warped[0] > x_end)
        if overlap:
            log("  WARNING: image2's artifact line overlaps image1's -- patch may be incomplete")
        else:
            log(f"  image2's own line at columns {line2_in_warped[0]}-{line2_in_warped[1]} (no overlap, good)")

    log("Matching local tone of patch to surrounding image...")
    warped2_toned = match_local_tone(img1_corrected, warped2, x_start, x_end)

    log("Blending with feather...")
    result = feather_patch(img1_corrected, warped2_toned, x_start, x_end, feather=feather)

    cv2.imwrite(output, result)
    log(f"Saved result: {output}")

    if debug:
        out_dir = os.path.dirname(os.path.abspath(output))
        save_debug_steps(out_dir, x_start, x_end, v_offset, [
            ("01_original",         img1,            "Image 1 before any processing"),
            ("02_vertical_fixed",   img1_corrected,  f"After {v_offset}px vertical sensor correction"),
            ("03_img2_aligned",     warped2,         "Image 2 after homography alignment to image 1"),
            ("04_img2_tone_matched", warped2_toned,  "Image 2 after local tone matching"),
            ("05_final_result",     result,          "Final blended result"),
        ], log=log)

    return {"output": output, "dpi": dpi, "line": (x_start, x_end), "vertical_offset": v_offset}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image1", help="first scan (base)")
    ap.add_argument("image2", nargs="?",
                    help="second scan (shifted to cover the artifact); not needed for --calibrate")
    ap.add_argument("-o", "--output", default="stitched_result.png")
    ap.add_argument("--dpi", type=int, default=None,
                    help="DPI profile to use (default: read from image header)")
    ap.add_argument("--rotate-deg", type=int, choices=[0, 90, 180, 270], default=0)
    ap.add_argument("--feather", type=int, default=40)
    ap.add_argument("--vertical-offset", type=int, default=None,
                    help="override saved sensor offset (pixels, positive = right side too low)")
    ap.add_argument("--calibrate", action="store_true",
                    help="detect line position + save config for this DPI, then exit")
    ap.add_argument("--debug", action="store_true",
                    help="save per-step debug images next to the output")
    args = ap.parse_args()

    try:
        if args.calibrate:
            calibrate(args.image1, dpi=args.dpi, vertical_offset=args.vertical_offset)
            return
        if args.image2 is None:
            ap.error("image2 is required for normal (non-calibrate) use")
        run_pipeline(args.image1, args.image2, args.output,
                     dpi=args.dpi, vertical_offset=args.vertical_offset,
                     feather=args.feather, rotate_deg=args.rotate_deg, debug=args.debug)
    except PipelineError as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
