# Example Calibration Patterns for Common Setups

This directory contains example configuration files for common camera and lens setups.

## IDS u3-3680xcp-c at 20cm Distance

### Example 1: 6mm Wide-Angle Lens (RECOMMENDED for 20cm)
**File:** `stereo_config_6mm_20cm.yaml`

Best for close-range work. Field of view: ~190mm wide at 20cm.

```yaml
system:
  baseline_estimate: 120.0
  calibration_pattern:
    type: "chessboard"
    width: 10      # internal corners
    height: 7
    square_size: 12.0  # mm
```

Pattern to generate:
```bash
python scripts/generate_pattern.py --rows 8 --cols 11 --size 12
```

### Example 2: 8mm Medium Lens
**File:** `stereo_config_8mm_20cm.yaml`

Good compromise. Field of view: ~143mm wide at 20cm.

```yaml
system:
  calibration_pattern:
    type: "chessboard"
    width: 9       # internal corners
    height: 6
    square_size: 10.0  # mm
```

Pattern to generate:
```bash
python scripts/generate_pattern.py --rows 7 --cols 10 --size 10
```

## For 30cm Distance (Easier)

### Example 3: 6mm Lens at 30cm
**File:** `stereo_config_6mm_30cm.yaml`

Easier to work with, more forgiving.

```yaml
system:
  calibration_pattern:
    type: "chessboard"
    width: 9       # internal corners
    height: 6
    square_size: 18.0  # mm
```

Pattern to generate:
```bash
python scripts/generate_pattern.py --rows 7 --cols 10 --size 18
```

## How to Use

1. **Determine your lens focal length** (check lens marking)
2. **Decide on calibration distance** (20cm, 30cm, etc.)
3. **Copy the appropriate example** to `stereo_config.yaml`
4. **Or run the calculator** for your specific setup:
   ```bash
   python scripts/calculate_calibration_pattern.py --focal 6 --distance 200
   ```
5. **Generate and print the pattern**
6. **Capture calibration images**
7. **Run calibration**

See `docs/CALIBRATION_PATTERN_GUIDE.md` for detailed instructions.
