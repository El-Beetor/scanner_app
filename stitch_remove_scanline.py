#!/usr/bin/env python3
"""
stitch_remove_scanline.py

Removes a fixed-column scanner sensor artifact (a thin vertical blue line) by
stitching two scans of the same page taken with a small horizontal shift, so
the columns lost to the artifact in scan 1 are clean in scan 2.

Pipeline (see run_pipeline):
  1. In each scan, measure the vertical step between the scanner's left and
     right sensor halves and shift everything right of the artifact column to
     undo it. The step is not a constant — at 1200dpi it changes from scan to
     scan — so it is measured per scan (measure_vertical_offset).
  2. Align scan 2 onto scan 1 (ORB features + RANSAC homography).
  3. Match the patch's brightness/contrast to the surrounding columns.
  4. Feather-blend the patch over the artifact columns.

The artifact column is a hardware constant for a given DPI: calibrate once per
DPI (--calibrate) and it is stored in scanner_config.json next to this script.
The profile also stores a default vertical offset, used only for scans too
blank near the line to measure.

Usage:
    # One-time per DPI: detect the line (use a scan of a mostly blank page)
    python3 stitch_remove_scanline.py blank.jpg --calibrate --dpi 600
    # optionally set the fallback vertical offset by hand
    python3 stitch_remove_scanline.py blank.jpg --calibrate --dpi 600 --vertical-offset 5

    # Normal use (DPI is read from the file's JFIF header, or pass --dpi)
    python3 stitch_remove_scanline.py scan1.jpg scan2.jpg -o result.png

    Optional flags:
      --dpi N                       DPI profile to use (default: read from file)
      --rotate-deg {0,90,180,270}   rotation to apply to scan 2 (default 0)
      --feather N                   blend-feather width in px (default 40)
      --vertical-offset N           force this vertical offset (px) for both scans
                                    instead of measuring it (0 = no correction)
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
    """Return the DPI profile name to use: the explicit value, else the file's header."""
    if dpi:
        return str(dpi)
    found = read_exif_dpi(image_path)
    if found:
        log(f"  DPI from file header: {found}")
        return str(found)
    raise PipelineError(
        f"could not read the DPI from {os.path.basename(image_path)}'s header; pass --dpi N")


def check_dpi_matches(image_path, img, dpi, profile):
    """Raise if the file's header DPI or its pixel width contradict the chosen profile.
    Guards against running a 1200dpi scan through the 600dpi line position (which
    would patch the wrong columns and leave the real line untouched)."""
    name = os.path.basename(image_path)
    hdr = read_exif_dpi(image_path)
    if hdr and str(hdr) != str(dpi):
        raise PipelineError(
            f"{name} header says {hdr}dpi but the {dpi}dpi profile was selected")
    expected = profile.get("image_width")
    if expected and img.shape[1] != expected:
        raise PipelineError(
            f"{name} is {img.shape[1]}px wide but the {dpi}dpi profile expects "
            f"{expected}px — wrong DPI selected?")


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


OFFSET_CONFIDENCE = 0.05   # measure_vertical_offset() confidence needed to trust a measurement


def _profile_shift(a, b, max_shift):
    """Rows by which 1-D gradient profile b sits lower than a (sub-pixel), plus the
    normalised-correlation peak and the best score away from that peak."""
    n = len(a)
    scores = np.empty(2 * max_shift + 1)
    for k, s in enumerate(range(-max_shift, max_shift + 1)):
        aa, bb = (a[:n - s], b[s:]) if s >= 0 else (a[-s:], b[:n + s])
        aa = aa - aa.mean()
        bb = bb - bb.mean()
        scores[k] = (aa * bb).sum() / (np.sqrt((aa * aa).sum() * (bb * bb).sum()) + 1e-9)
    i = int(np.argmax(scores))
    s = float(i - max_shift)
    if 0 < i < len(scores) - 1:                       # parabolic sub-pixel refinement
        y0, y1, y2 = scores[i - 1], scores[i], scores[i + 1]
        den = y0 - 2 * y1 + y2
        if den != 0:
            s += float(np.clip((y0 - y2) / (2 * den), -0.5, 0.5))
    runner = scores[np.abs(np.arange(len(scores)) - i) > 2].max()
    return s, float(scores[i]), float(runner)


def measure_vertical_offset(img, x_start, x_end, max_shift=40):
    """
    Measure, for THIS scan, how many rows the content right of the artifact line
    sits lower than the content left of it (the scanner's inter-sensor step).
    It is not a constant: at 1200dpi it varies from scan to scan.

    Each side is collapsed to a 1-D "horizontal-edge" profile (column-averaged
    intensity, differentiated along y) and the two are cross-correlated over
    ±max_shift rows. Page skew would masquerade as a step, so the same shift is
    measured between two strips on the *same* side (skew only) and subtracted.

    Returns (offset_px, confidence). confidence is the weakest correlation-peak
    margin of the three measurements; >= OFFSET_CONFIDENCE is trustworthy, below
    it the scan has too little horizontal detail near the line.
    """
    h, w = img.shape[:2]
    lw = x_end - x_start + 1
    gap, strip = lw, 2 * lw                  # skip the colour fringe, then average 2 line-widths
    D = lw + 2 * gap + strip                 # centre-to-centre distance of the across-line pair
    x0 = x_start - gap - strip - D
    x1 = x_end + 1 + gap + D + strip
    if x0 < 0 or x1 > w:
        return 0, 0.0
    band = cv2.cvtColor(img[int(h * 0.10):, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)

    def prof(x):                                     # x in image coordinates
        return np.diff(band[:, x - x0: x - x0 + strip].mean(axis=1))

    l_in, r_in = prof(x_start - gap - strip), prof(x_end + 1 + gap)
    l_out, r_out = prof(x_start - gap - strip - D), prof(x_end + 1 + gap + D)
    across = _profile_shift(l_in, r_in, max_shift)   # step + skew
    left = _profile_shift(l_out, l_in, max_shift)    # skew only
    right = _profile_shift(r_in, r_out, max_shift)   # skew only

    step = across[0] - (left[0] + right[0]) / 2
    conf = min(peak - runner for _, peak, runner in (across, left, right))
    if any(abs(s) >= max_shift for s, _, _ in (across, left, right)):
        conf = 0.0                                   # hit the search boundary: not a real peak
    return int(round(step)), conf


def resolve_vertical_offset(img, forced, profile, x_start, x_end, label, log=print):
    """Offset to correct `img` with: the forced value if given, else this scan's
    measurement if confident, else the profile's saved default."""
    if forced is not None:
        log(f"  {label}: vertical offset forced to {forced:+d}px")
        return forced
    measured, conf = measure_vertical_offset(img, x_start, x_end)
    if conf >= OFFSET_CONFIDENCE:
        log(f"  {label}: measured vertical offset {measured:+d}px (confidence {conf:.2f})")
        return measured
    default = profile.get("vertical_offset", 0)
    log(f"  {label}: could not measure the vertical offset reliably (confidence {conf:.2f}); "
        f"using the profile default {default:+d}px")
    return default


# ── Pipeline stages ───────────────────────────────────────────────────────────

def apply_vertical_offset(img, x_split, offset):
    """
    Corrects the scanner's inter-sensor vertical offset by shifting everything
    to the right of x_split up (offset > 0) or down (offset < 0) by |offset| px.
    """
    if offset == 0:
        return img

    h = img.shape[0]
    result = img.copy()                      # reads come from img, so no aliasing
    if offset > 0:                           # right side sits too low: move it up
        result[:h - offset, x_split:] = img[offset:, x_split:]
        result[h - offset:, x_split:] = img[-1:, x_split:]
    else:                                    # right side sits too high: move it down
        o = -offset
        result[o:, x_split:] = img[:h - o, x_split:]
        result[:o, x_split:] = img[:1, x_split:]
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

    # Only the artifact columns change, so only they are taken to float.
    roi = patch[:, x_start:x_end + 1].astype(np.float32)
    for c in range(3):
        ref_mean, ref_std = ref_cols[:, c].mean(), ref_cols[:, c].std() + 1e-6
        src_mean, src_std = patch_cols[:, c].mean(), patch_cols[:, c].std() + 1e-6
        scale = ref_std / src_std
        shift_val = ref_mean - src_mean * scale
        roi[:, :, c] = patch[:, x_start:x_end + 1, c].astype(np.float32) * scale + shift_val

    out = patch.copy()
    out[:, x_start:x_end + 1] = np.clip(roi, 0, 255).astype(np.uint8)
    return out


def feather_patch(base, patch, x_start, x_end, feather=40):
    """Replace columns [x_start, x_end] of base with patch, ramping alpha over
    `feather` px on each side so there is no hard seam."""
    h, w = base.shape[:2]
    fx_start = max(0, x_start - feather)
    fx_end = min(w - 1, x_end + feather)

    def alpha_at(x):
        if x < x_start:
            return (x - fx_start) / max(1, (x_start - fx_start))
        if x > x_end:
            return 1 - (x - x_end) / max(1, (fx_end - x_end))
        return 1.0

    # Only the feather window [fx_start, fx_end] changes, so only it is taken to
    # float (the full image would be ~1.7 GB per copy at 1200dpi).
    alpha = np.array([alpha_at(x) for x in range(fx_start, fx_end + 1)])
    a = alpha.astype(np.float32)[None, :, None]
    b = (1 - alpha).astype(np.float32)[None, :, None]
    base_f = base[:, fx_start:fx_end + 1].astype(np.float32)
    patch_f = patch[:, fx_start:fx_end + 1].astype(np.float32)
    blended = b * base_f + a * patch_f

    out = base.copy()
    out[:, fx_start:fx_end + 1] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def save_debug_steps(out_dir, x_start, x_end, steps, log=print):
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
    check_dpi_matches(image1, img1, dpi, profile)

    have_line = "line_start" in profile and "line_end" in profile
    if have_line and vertical_offset is not None:
        # Only setting the default offset on an existing profile: keep the line as is,
        # since the image given here may not be a good one to detect it on.
        x_start, x_end = profile["line_start"], profile["line_end"]
        log(f"Keeping saved line position: columns {x_start}-{x_end}")
    else:
        log("Detecting artifact line...")
        line = detect_blue_line(img1)
        if line is None:
            raise PipelineError("could not detect a blue artifact line — calibrate on a "
                                "scan of a mostly blank, non-blue page")
        x_start, x_end = line
        log(f"  found at columns {x_start}-{x_end} (width {x_end - x_start + 1}px)")
        if have_line and (x_start, x_end) != (profile["line_start"], profile["line_end"]):
            log(f"  (replaces previously saved {profile['line_start']}-{profile['line_end']})")

    # The step is measured per scan at stitch time; the profile only stores a
    # default for scans that are too blank near the line to measure.
    if vertical_offset is not None:
        v_offset = vertical_offset
        log(f"Default vertical offset set to {v_offset:+d}px")
    else:
        measured, conf = measure_vertical_offset(img1, x_start, x_end)
        if conf >= OFFSET_CONFIDENCE:
            v_offset = measured
            log(f"Measured vertical offset on this scan: {measured:+d}px (confidence {conf:.2f}) "
                f"— saved as the profile default")
        else:
            v_offset = profile.get("vertical_offset", 0)
            log(f"Could not measure the vertical offset reliably on this scan (confidence "
                f"{conf:.2f}); default stays {v_offset:+d}px — pass --vertical-offset N to set it")

    profile["line_start"] = x_start
    profile["line_end"] = x_end
    profile["vertical_offset"] = v_offset
    profile["image_width"] = img1.shape[1]
    cfg[dpi_key(dpi)] = profile
    save_config(cfg, config_path, log)
    log(f"Calibration saved for DPI profile '{dpi}': "
        f"line={x_start}-{x_end}, vertical_offset={v_offset}px, image_width={img1.shape[1]}px")
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

    check_dpi_matches(image1, img1, dpi, profile)

    img2 = load_image(image2)
    rot_map = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
               180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    img2_oriented = img2 if rot_map[rotate_deg] is None else cv2.rotate(img2, rot_map[rotate_deg])
    check_dpi_matches(image2, img2_oriented, dpi, profile)
    if img2_oriented.shape[:2] != img1.shape[:2]:
        # Same scanner + same DPI + same scan area always gives identical sizes;
        # anything else means the line column is wrong too, so don't resize-and-hope.
        raise PipelineError(
            f"scan sizes differ: {os.path.basename(image1)} is {img1.shape[1]}x{img1.shape[0]}, "
            f"{os.path.basename(image2)} is {img2_oriented.shape[1]}x{img2_oriented.shape[0]} — "
            "both scans must use the same DPI and scan area")

    # The scanner's left/right sensor halves are vertically out of step, and the
    # step differs from scan to scan, so each scan is measured and corrected on
    # its own. Scan 2 matters too: the patch comes from its right half, and an
    # uncorrected step there makes the homography split the difference.
    log("Measuring vertical sensor step in each scan...")
    v1 = resolve_vertical_offset(img1, vertical_offset, profile, x_start, x_end, "scan 1", log)
    if rotate_deg == 0:
        v2 = resolve_vertical_offset(img2_oriented, vertical_offset, profile, x_start, x_end, "scan 2", log)
    else:
        v2 = 0
        log("  scan 2: rotated, so its line column is unknown — not corrected")
    img1_corrected = apply_vertical_offset(img1, x_start, v1)
    img2_corrected = apply_vertical_offset(img2_oriented, x_start, v2)

    log("Aligning image2 onto image1's frame...")
    warped2 = align_to_base(img1_corrected, img2_corrected, log=log)
    del img2, img2_oriented, img2_corrected      # ~0.4 GB each at 1200dpi; no longer needed

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
        save_debug_steps(out_dir, x_start, x_end, [
            ("01_original",         img1,            "Image 1 before any processing"),
            ("02_vertical_fixed",   img1_corrected,  f"Image 1 after {v1:+d}px vertical sensor correction"),
            ("03_img2_aligned",     warped2,         f"Image 2 after {v2:+d}px correction + homography alignment"),
            ("04_img2_tone_matched", warped2_toned,  "Image 2 after local tone matching"),
            ("05_final_result",     result,          "Final blended result"),
        ], log=log)

    return {"output": output, "dpi": dpi, "line": (x_start, x_end),
            "vertical_offset": v1, "vertical_offset_scan2": v2}


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
