# scan-line-fix

Removes a fixed-column scanner/camera sensor artifact (e.g. a thin colored
vertical line) from a scanned document, using a second scan of the same
page taken in a different physical orientation.

## The problem

Some scanners and phone scanning apps leave a thin, strongly-colored
vertical line at a **fixed pixel column** in every image they produce — a
sensor defect, light leak, or processing artifact tied to the device, not
the page. If you scan a page once, then physically rotate the page (e.g.
180°) and scan it again, the artifact stays at the same column in the
file, but now lands on different page content, since the page rotated
underneath it.

That gives you two scans where the corrupted column differs — meaning
each scan has clean data exactly where the other one doesn't.

## How it works

1. **Rotate** the second scan to match the first scan's orientation
   (default 180°, configurable for 90°/270° as well).
2. **Align** the two scans precisely using ORB feature matching +
   homography (RANSAC). Real-world scans/photos are rarely
   pixel-perfect registered even after the rotation is corrected, so
   this step accounts for any residual shift, skew, or perspective
   difference between the two passes.
3. **Detect** the artifact's column range in each image, by scoring how
   much more one color channel (blue, by default) dominates the other
   two in each column compared to the image's background.
4. **Patch**: replace the artifact's column range in image 1 with the
   same columns from the aligned image 2, with a feathered blend at the
   edges so there's no visible seam.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 stitch_remove_scanline.py image1.jpg image2.jpg -o result.jpg
```

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `-o, --output` | `stitched_result.jpg` | Output file path |
| `--rotate-deg` | `180` | Rotation (`0`/`90`/`180`/`270`) to apply to image2 so it matches image1's orientation |
| `--feather` | `15` | Blend feather width in pixels at the patch edges |
| `--debug` | off | Also saves `debug_detected_line.jpg` (detected line region boxed in red on image1) and `debug_warped_image2.jpg` (the aligned second scan) — useful for sanity-checking detection/alignment on a new scan pair |

## How detection works

The script looks for the image column where one color channel (blue, by
default) dominates the other two far more than anywhere else in the
image — characteristic of a thin, strongly-colored sensor artifact rather
than actual drawn or printed content. This works well for artifacts that
are highly saturated and only a few pixels wide. It's tuned for a blue
line specifically (`detect_blue_line()`), but the channel logic is only a
few lines to adapt if your artifact is a different color.

## Limitations

- Assumes the artifact is a thin (a few px), strongly color-dominant
  vertical line — not suited to wide bands, faint streaks, or artifacts
  that blend into similarly-colored page content.
- Alignment quality depends on having enough visual texture/detail for
  ORB to find good keypoints; a nearly blank page may not align well
  (the script falls back to phase-correlation translation-only alignment
  in that case, and prints a warning when it does).
- Only patches *one* image's artifact column (image1, by default) in the
  single merged output. If you need both orientations cleaned up, run it
  twice, swapping which image is passed first.

## License

MIT — see [LICENSE](LICENSE).
