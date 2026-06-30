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
  2. Aligns the two scans precisely (ORB feature matching + homography),
     since real-world scans/photos are rarely pixel-perfect registered.
  3. Detects the artifact line's column range in each image.
  4. Produces one merged image, using image 1 as the base, with its
     artifact-line columns replaced by the (clean, aligned) equivalent
     columns from image 2 -- feathered at the edges to avoid a hard seam.

Usage:
    python3 stitch_remove_scanline.py img1.jpg img2.jpg -o result.jpg

    Optional flags:
      --rotate-deg {0,90,180,270}   rotation needed to match img2 to img1 (default 180)
      --feather N                   blend-feather width in px on each side (default 15)
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
    """
    Find a thin, strongly-blue vertical line (column range) in an image,
    by looking for the column(s) where blue dominates red/green far more
    than anywhere else in the image.

    Returns (x_start, x_end) inclusive column range, or None if no clear
    line is found.
    """
    b, g, r = cv2.split(img.astype(np.int16))
    blueness = np.clip(b - np.maximum(r, g), 0, None)
    col_score = blueness.sum(axis=0).astype(np.float64)

    # smooth slightly so a 2-4px-wide line registers as one coherent peak
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
    """
    Estimate a homography mapping moving_img onto base_img's frame and warp it.
    Falls back to translation-only (phase correlation) if not enough features
    are found, and to no alignment at all (with a warning) if that also fails.
    Returns the warped image, same size as base_img.
    """
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

    result = feather_patch(img1, warped2, line1[0], line1[1], feather=args.feather)
    cv2.imwrite(args.output, result)
    print(f"Saved result: {args.output}")

    if args.debug:
        debug = img1.copy()
        cv2.rectangle(debug, (line1[0], 0), (line1[1], debug.shape[0] - 1), (0, 0, 255), 3)
        cv2.imwrite("debug_detected_line.jpg", debug)
        cv2.imwrite("debug_warped_image2.jpg", warped2)
        print("Saved debug images: debug_detected_line.jpg, debug_warped_image2.jpg")


if __name__ == "__main__":
    main()
