# scanner_app — remove a fixed-column scanner artifact

Some flatbed scanners leave a thin, strongly coloured vertical line at a fixed
pixel column of every scan (here: a blue line from an Epson unit, ~25px wide at
600dpi, ~43px at 1200dpi). This tool removes it losslessly by combining two
scans of the same page.

## How to scan

1. Scan the page as usual (**scan 1**).
2. Slide the page sideways by a small amount — more than the line width plus the
   40px blend margin, a quarter inch is plenty — without rotating it, and scan
   again (**scan 2**). Same DPI, same scan area.
3. Stitch. Scan 1 is the base; the columns under its artifact are replaced by
   the same content from scan 2, where the line fell somewhere else.

Keep the page square to the bed. The old flip-the-page-180° method still works
with `--rotate-deg 180`, but the sideways shift keeps lighting and geometry
identical between the two scans and aligns far better.

## Install

```bash
pip install -r requirements.txt
```

With Homebrew Python on macOS, Tk is a separate package: `brew install python-tk@3.14`
(match your Python version). Only the GUI needs it.

## Calibrate once per DPI

The artifact column is a hardware constant for a given DPI, so it is detected
once and saved, never re-detected during normal use (blue artwork could fool
the detector). Use a scan of a mostly blank page:

```bash
python3 stitch_remove_scanline.py blank.jpg --calibrate --dpi 600
```

This stores the line columns, the scan's pixel width, and a default vertical
offset (see below) in `scanner_config.json` next to the script, under
`dpi_600`. Repeat for each DPI you scan at. The config is part of the repo on
purpose — it *is* the scanner's calibration.

## Use

Command line:

```bash
python3 stitch_remove_scanline.py scan1.jpg scan2.jpg -o result.png
```

GUI (drag-and-drop):

```bash
python3 scanner_ui.py
```

Drop the two scans, check the DPI it picked up from the file, click Run. The
output name defaults to `<scan1>_stitched.png` next to scan 1.

Output is PNG by default — lossless, so the stitch adds no compression
artifacts on top of the scanner's JPEG.

| Flag | Default | Description |
|---|---|---|
| `-o, --output` | `stitched_result.png` | Output path |
| `--dpi N` | from file header | Which calibration profile to use. An explicit value wins over a wrong header (edited files often say 72dpi); the scan's pixel width must match the profile either way |
| `--rotate-deg {0,180}` | `0` | `180` for the legacy flip-the-page method |
| `--feather N` | `40` | Blend ramp width in px on each side of the patch |
| `--vertical-offset N` | measured | Force the sensor-step correction for both scans instead of measuring it (`0` = none) |
| `--calibrate` | | Detect the line on `scan1` and save the profile for this DPI, then exit |
| `--force` | | With `--calibrate`: accept a detected line far from the saved one |
| `--debug` | | Save `debug_01…05_{full,seam}.jpg` next to the output, one per pipeline step |

## What it does

1. **Vertical sensor step.** The scanner's left and right sensor halves are
   vertically out of register at the artifact column: everything to the right
   sits a few rows higher or lower. This is *not* a constant — at 600dpi it is
   reliably +5px, at 1200dpi it changes from scan to scan (−9 to +2 measured on
   the same page minutes apart). So each scan is measured individually:
   both sides of the line are collapsed to a 1-D horizontal-edge profile,
   cross-correlated over ±40 rows, and page skew is cancelled by subtracting
   the same measurement taken on one side only. Both scans are then corrected
   with a whole-row shift (no resampling). If a scan has too little — or too
   repetitive — horizontal detail near the line to measure, the profile's
   default is used and a warning is reported.
2. **Alignment.** Scan 2 is warped onto scan 1 with an ORB + RANSAC homography.
   The known column of scan 2's own artifact is pushed through that homography
   to make sure it landed clear of the patch; on top of scan 1's line it is an
   error (the patch would copy the artifact back in), inside the blend margin a
   warning.
3. **Tone match.** The patch's per-channel mean/contrast is matched to the
   columns either side of the line.
4. **Feather blend.** The patch replaces the artifact columns with a linear
   alpha ramp over `--feather` px on each side.

The pipeline is deterministic, so a saved output is a valid regression
baseline (`cmp`).

## Limitations

- Needs two scans of the same page; nothing is invented or inpainted.
- The artifact must be a vertical band of strongly blue pixels for
  calibration to find it (`detect_blue_line`); a different colour is a few
  lines to change.
- Rotations of 90°/270° are not supported: scan 2's artifact would become a
  horizontal band no column patch can avoid.
- Nearly blank pages, ruled paper and coarse halftones near the line can
  defeat the per-scan step measurement; the profile default is used then.
- Memory: about 2.4 GB peak for a 1200dpi letter-size pair (143 MP).

## License

MIT — see [LICENSE](LICENSE).
