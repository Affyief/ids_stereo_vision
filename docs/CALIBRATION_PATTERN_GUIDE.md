# Calibration Pattern Guide for IDS u3-3680xcp-c at Close Range

## Quick Answer for 20cm Distance

**You need to know your lens focal length first!** Check the side of your C-mount lens for markings (e.g., "6mm", "8mm", "12mm").

### For Common Lenses at 20cm on A4 Paper:

#### 6mm Lens (Wide Angle) - RECOMMENDED for 20cm
- **Pattern**: 8×6 corners (9×7 squares)
- **Square size**: 12mm
- **Print size**: 108mm × 84mm
- **Fits easily on A4**: ✓ YES
- **Field of view**: ~190mm wide at 20cm

```yaml
# config/stereo_config.yaml
width: 8
height: 6
square_size: 12.0
```

#### 8mm Lens (Medium)
- **Pattern**: 7×5 corners (8×6 squares)
- **Square size**: 10mm
- **Print size**: 80mm × 60mm
- **Fits easily on A4**: ✓ YES
- **Field of view**: ~143mm wide at 20cm

```yaml
# config/stereo_config.yaml
width: 7
height: 5
square_size: 10.0
```

## How to Use

### Step 1: Find Your Lens Focal Length
Look at your C-mount lens. Common values: 6mm, 8mm, 12mm

### Step 2: Calculate Pattern
```bash
python scripts/calculate_calibration_pattern.py --focal 6 --distance 200
```

### Step 3: Generate Pattern
```bash
python scripts/generate_pattern.py --rows 7 --cols 9 --size 12
```

### Step 4: Print & Use
Print at 100% scale (no fit-to-page), verify with ruler, then calibrate!

See full documentation in this file for details.
