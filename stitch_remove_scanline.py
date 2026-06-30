#!/usr/bin/env python3
"""
stitch_remove_scanline.py

Fixes a thin vertical scanner/sensor artifact (e.g. a blue line) that shows up
at a fixed pixel column on every scan from a given device. If you scan the
SAME page twice in two different physical orientations (e.g. one scan, then
flip the page 180 degrees and scan again), the artifact stays at the same
column in the file -- but lands on different content each time, since the
page rotated underneath it.

This script:
  1. Rotates the second image to match the first's orientation.
  2. Aligns the two scans precisely (ORB feature matching + homography).
  3. Detects the artifact line's column range in each image.
  4. Matches local brightness/contrast of the patch to surrounding columns.
  5. Blends using Poisson seamless cloning to hide the seam.
  6. Falls back to feathered blending if Poisson fails.

Usage:
    python3 stitch_remove_scanline.py img1.jpg img2.jpg -o result.jpg

    Optional flags:
      --rotate-deg {0,90,180,270}   rotation needed to match img2 to img1 (default 180)
      --feather N                   blend-feather width in px on each side (default 15)
      --blend {poisson,feather}     blending method (default poisson)
      --debug                       also save line-detection + alignment-diff debug images
"""

import argparse
import sys
import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        sys.exit(f"ERROR: could not read image: {path}")
    return img


def detect_blue_line(img, min_strength=500):
    b, g, r = cv2.split(img.astype(np.int16))
    blueness = np.clip(b - np.maximum(r, g), 0, None)
    col_score = blueness.sum(axis=0).astype(np.float64)

    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(col_score, kernel, mode="same")

    peak_col = int(np.argmax(smoothed))
    peak_val = smoothed[peak_col]
    background = np.median(smoothed)

    if peak_val < min_strength or peak_val < background * 20:
        return None

    threshold = background + (peak_val - background) * 0.12
    x_start = peak_col
    while x_start > 0 and smoothed[x_start - 1] > threshold:
        x_start -= 1
    x_end = peak_col
    while x_end < len(smoothed) - 1 and smoothed[x_end + 1] > threshold:
        x_end += 1

    return (x_start, x_end)


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
    """
    After global homography alignment there can still be local warp differences
    (paper curl, perspective) right at the seam. This computes dense optical
    flow between base and patch in a band around the artifact and applies that
    local displacement to the patch before blending, giving a much tighter fit.
    """
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

    roi_patch = patch[:, rx0:rx1]
    warped_roi = cv2.remap(roi_patch, map_x, map_y,
                           interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

    result = patch.copy()
    result[:, rx0:rx1] = warped_roi
    return result


def match_local_tone(base, patch, x_start, x_end, context_width=60):
    """
    Adjust patch columns so their mean and std match the surrounding columns
    in base. This corrects any exposure or contrast difference between the
    two scans before blending.
    """
    h, w = base.shape[:2]
    left = max(0, x_start - context_width)
    right = min(w, x_end + 1 + context_width)

    # sample the base columns either side of the artifact (excluding the artifact itself)
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
        shift = ref_mean - src_mean * scale
        adjusted[:, x_start:x_end + 1, c] = (
            patch[:, x_start:x_end + 1, c].astype(np.float32) * scale + shift
        )

    return np.clip(adjusted, 0, 255).astype(np.uint8)


def poisson_blend(base, patch, x_start, x_end, feather=15):
    """
    Use Poisson seamless cloning to blend the patch strip into the base.
    Operates only on a padded region around the artifact to keep it fast.
    Falls back to feather blending if seamlessClone fails.
    """
    h, w = base.shape[:2]
    pad = feather + 10
    rx0 = max(0, x_start - pad)
    rx1 = min(w, x_end + 1 + pad)

    roi_base = base[:, rx0:rx1].copy()
    roi_patch = patch[:, rx0:rx1].copy()
    roi_w = rx1 - rx0

    # mask: white only over the artifact columns (relative to the ROI)
    mask = np.zeros((h, roi_w), dtype=np.uint8)
    rel_start = x_start - rx0
    rel_end = x_end - rx0
    mask[:, rel_start:rel_end + 1] = 255

    center = (roi_w // 2, h // 2)

    try:
        blended_roi = cv2.seamlessClone(roi_patch, roi_base, mask, center, cv2.NORMAL_CLONE)
        result = base.copy()
        result[:, rx0:rx1] = blended_roi
        return result
    except cv2.error as e:
        print(f"  WARNING: Poisson blending failed ({e}), falling back to feather blend")
        return feather_patch(base, patch, x_start, x_end, feather)


def feather_patch(base, patch, x_start, x_end, feather=15):
    """Replace columns [x_start, x_end] of base with patch's columns, feathering the
    transition over `feather` px on each side so there's no hard seam."""
    out = base.astype(np.float32).copy()
    patch = patch.astype(np.float32)
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
        out[:, x] = (1 - alpha) * out[:, x] + alpha * patch[:, x]

    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image1", help="path to the first scan (used as the base orientation)")
    ap.add_argument("image2", help="path to the second scan (rotated relative to the first)")
    ap.add_argument("-o", "--output", default="stitched_result.jpg", help="output file path")
    ap.add_argument("--rotate-deg", type=int, choices=[0, 90, 180, 270], default=180,
                     help="rotation to apply to image2 so it matches image1's orientation")
    ap.add_argument("--feather", type=int, default=15, help="blend feather width in pixels")
    ap.add_argument("--blend", choices=["poisson", "feather"], default="poisson",
                     help="blending method: poisson (default, best quality) or feather")
    ap.add_argument("--debug", action="store_true", help="save extra debug images")
    args = ap.parse_args()

    img1 = load_image(args.image1)
    img2 = load_image(args.image2)

    rot_map = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    img2_oriented = img2 if rot_map[args.rotate_deg] is None else cv2.rotate(img2, rot_map[args.rotate_deg])

    if img2_oriented.shape[:2] != img1.shape[:2]:
        img2_oriented = cv2.resize(img2_oriented, (img1.shape[1], img1.shape[0]))

    print("Aligning image2 onto image1's frame...")
    warped2 = align_to_base(img1, img2_oriented)

    print("Detecting artifact line in image1...")
    line1 = detect_blue_line(img1)
    if line1 is None:
        sys.exit("ERROR: could not detect a blue artifact line in image1 -- adjust detect_blue_line() thresholds")
    print(f"  found at columns {line1[0]}-{line1[1]} (width {line1[1]-line1[0]+1}px)")

    line2_in_warped = detect_blue_line(warped2)
    if line2_in_warped is not None:
        overlap = not (line2_in_warped[1] < line1[0] or line2_in_warped[0] > line1[1])
        if overlap:
            print("  WARNING: image2's artifact line overlaps image1's after alignment -- "
                  "patch may not fully remove the artifact in the overlapping rows/columns")
        else:
            print(f"  image2's own line is at columns {line2_in_warped[0]}-{line2_in_warped[1]} (no overlap, good)")

    print("Applying local optical-flow warp to patch region...")
    warped2_local = local_warp_patch(img1, warped2, line1[0], line1[1])

    print("Matching local tone of patch to surrounding image...")
    warped2_toned = match_local_tone(img1, warped2_local, line1[0], line1[1])

    print(f"Blending with method: {args.blend}...")
    if args.blend == "poisson":
        result = poisson_blend(img1, warped2_toned, line1[0], line1[1], feather=args.feather)
    else:
        result = feather_patch(img1, warped2_toned, line1[0], line1[1], feather=args.feather)

    cv2.imwrite(args.output, result)
    print(f"Saved result: {args.output}")

    if args.debug:
        debug = img1.copy()
        cv2.rectangle(debug, (line1[0], 0), (line1[1], debug.shape[0] - 1), (0, 0, 255), 3)
        cv2.imwrite("debug_detected_line.jpg", debug)
        cv2.imwrite("debug_warped_image2.jpg", warped2)
        cv2.imwrite("debug_local_warped.jpg", warped2_local)
        cv2.imwrite("debug_toned_patch.jpg", warped2_toned)
        print("Saved debug images: debug_detected_line.jpg, debug_warped_image2.jpg, debug_toned_patch.jpg")


if __name__ == "__main__":
    main()
