# Solution Summary: Calibration Pattern for 20cm Distance

## Your Question
> "for this exact cameras and lens that i have, and i wanna do calib at 20cm distance, what's the best calib pattern? on a A4 paper sheet"

## The Answer

**The best calibration pattern depends on your lens focal length!**

### Most Common Configuration (6mm Lens) ⭐
If you have a **6mm wide-angle lens** (most common for close work):

```yaml
Pattern: 10×7 corners (11×8 squares)
Square size: 12mm
Total size: 132mm × 96mm
```

**To use this:**
```bash
cp config/stereo_config_6mm_20cm.yaml config/stereo_config.yaml
python scripts/generate_pattern.py --rows 8 --cols 11 --size 12
```

### If You Have an 8mm Lens
```yaml
Pattern: 9×6 corners (10×7 squares)  
Square size: 10mm
Total size: 100mm × 70mm
```

**To use this:**
```bash
cp config/stereo_config_8mm_20cm.yaml config/stereo_config.yaml
python scripts/generate_pattern.py --rows 7 --cols 10 --size 10
```

## How to Determine Your Exact Pattern

Since I don't know your exact lens, I've created tools to calculate it:

### Step 1: Find Your Lens Focal Length
Look at your C-mount lens for markings like:
- "6mm F1.4"
- "8mm"
- "12mm 1:1.4"

### Step 2: Run the Calculator
```bash
python scripts/calculate_calibration_pattern.py --focal YOUR_FOCAL --distance 200
```

This will analyze your specific setup and recommend the optimal pattern!

### Step 3: Generate and Print
Use the recommended values from the calculator:
```bash
python scripts/generate_pattern.py --rows X --cols Y --size Z
```

Then print at **100% scale** on A4 paper.

## What Was Created for You

I've added several tools and configurations to help you:

### 1. Tools
- **Calculator** (`scripts/calculate_calibration_pattern.py`) - Analyzes your setup
- **Generator** (`scripts/generate_pattern.py`) - Creates printable patterns

### 2. Pre-configured Examples
- `config/stereo_config_6mm_20cm.yaml` - For 6mm lens at 20cm ⭐
- `config/stereo_config_8mm_20cm.yaml` - For 8mm lens at 20cm
- `config/stereo_config_6mm_30cm.yaml` - For 6mm lens at 30cm (easier!)

### 3. Documentation
- `QUICKSTART_20CM.md` - Step-by-step instructions ← **START HERE**
- `docs/CALIBRATION_PATTERN_GUIDE.md` - Detailed guide
- `config/PATTERN_EXAMPLES.md` - Example configurations

## Quick Start

If you're not sure about your lens, **most C-mount lenses for close work are 6mm**:

```bash
# Copy the 6mm configuration
cp config/stereo_config_6mm_20cm.yaml config/stereo_config.yaml

# Generate the pattern
python scripts/generate_pattern.py --rows 8 --cols 11 --size 12

# Print calibration_pattern.png at 100% scale

# Capture images
python scripts/capture_calibration_images.py --count 30

# Run calibration
python scripts/run_calibration.py --visualize
```

## Alternative: Move to 30cm (Easier!)

If 20cm is challenging, consider using 30cm distance instead:
- Larger pattern (easier to print)
- More forgiving
- Still good calibration quality

```bash
cp config/stereo_config_6mm_30cm.yaml config/stereo_config.yaml
python scripts/generate_pattern.py --rows 7 --cols 10 --size 18
```

## Need Help?

Read `QUICKSTART_20CM.md` for complete step-by-step instructions!
