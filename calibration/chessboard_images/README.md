# Calibration Images Directory

This directory stores calibration images captured from the cameras.

## Structure

- `left/` - Images from the left camera
- `right/` - Images from the right camera

## Usage

1. Run the capture script:
   ```bash
   python scripts/capture_calibration_images.py --count 30
   ```

2. Images will be saved as:
   - `left/calib_000.png`, `left/calib_001.png`, ...
   - `right/calib_000.png`, `right/calib_001.png`, ...

3. After capturing images, run calibration:
   ```bash
   python scripts/run_calibration.py
   ```

## Requirements

- Print a chessboard pattern (9x6 internal corners, 25mm squares by default)
- Capture at least 20-30 image pairs
- Vary the angle and distance of the chessboard
- Ensure the entire pattern is visible in both cameras
- Avoid motion blur

## Notes

Image files (*.png, *.jpg) in the left/ and right/ subdirectories are ignored by git.
Only the directory structure is tracked.
