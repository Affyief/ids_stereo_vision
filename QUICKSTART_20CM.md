# QUICK START: Calibration at 20cm for IDS u3-3680xcp-c

## Step 1: Find Your Lens Focal Length

Look at the side of your C-mount lens. You should see something like:
- "6mm F1.4"
- "8mm"  
- "f=6mm"
- "12mm 1:1.4"

The number before "mm" is your **focal length**. Common values: **6mm**, **8mm**, **12mm**

## Step 2: Use the Calculator

```bash
cd ids_stereo_vision

# Replace 6 with YOUR focal length
python scripts/calculate_calibration_pattern.py --focal 6 --distance 200
```

This will output:
- Your field of view at 20cm
- Top 5 recommended patterns
- Best pattern marked with ⭐
- Configuration to copy

## Step 3: Generate the Pattern

From the calculator output, use the recommended values:

```bash
# Example for 6mm lens (adjust based on YOUR calculator output)
python scripts/generate_pattern.py --rows 8 --cols 11 --size 12
```

This creates:
- `calibration_pattern.png` - Print this!
- `calibration_pattern_reference.png` - For verification

## Step 4: Print the Pattern

**CRITICAL SETTINGS:**
1. Open `calibration_pattern.png`
2. Print at **100% scale** (actual size)
3. Do NOT use "Fit to page"
4. Quality: Highest/Best
5. Paper: A4

**After printing:**
- Use a ruler to measure one square
- Should match exactly (e.g., 12mm)
- If wrong, adjust printer settings and reprint
- Mount on cardboard for flatness

## Step 5: Update Configuration

Copy the pattern settings to your config file:

```bash
# If using the example config (6mm lens)
cp config/stereo_config_6mm_20cm.yaml config/stereo_config.yaml

# OR manually edit
nano config/stereo_config.yaml
```

Update these lines with calculator output:
```yaml
system:
  calibration_pattern:
    width: 10      # from calculator
    height: 7      # from calculator
    square_size: 12.0  # from calculator
```

## Step 6: Capture & Calibrate

```bash
# Capture 30 calibration images
python scripts/capture_calibration_images.py --count 30

# Run calibration
python scripts/run_calibration.py --visualize

# Run the system!
python scripts/run_stereo_system.py
```

## Common Lens Configurations

### 6mm Lens at 20cm ⭐ RECOMMENDED
- Field of view: ~190mm
- Pattern: 10×7 corners, 12mm squares
- Easy to use, fits well

```bash
cp config/stereo_config_6mm_20cm.yaml config/stereo_config.yaml
python scripts/generate_pattern.py --rows 8 --cols 11 --size 12
```

### 8mm Lens at 20cm
- Field of view: ~143mm  
- Pattern: 9×6 corners, 10mm squares
- Good for closer work

```bash
cp config/stereo_config_8mm_20cm.yaml config/stereo_config.yaml
python scripts/generate_pattern.py --rows 7 --cols 10 --size 10
```

### 6mm Lens at 30cm (EASIER for beginners)
- Field of view: ~285mm
- Pattern: 9×6 corners, 18mm squares
- Larger pattern, easier to print

```bash
cp config/stereo_config_6mm_30cm.yaml config/stereo_config.yaml
python scripts/generate_pattern.py --rows 7 --cols 10 --size 18
```

## Troubleshooting

### "I don't know my focal length"
- **Most common for close work: 6mm or 8mm**
- Try both configurations and see which fits better
- If entire pattern fits easily in view → probably 6mm
- If pattern barely fits → probably 8mm or higher

### "Pattern doesn't fit in camera view"
- Your lens has higher focal length (12mm+)
- Move to 30cm distance, or
- Run calculator with your focal length
- Use smaller squares or fewer squares

### "Squares are too small to print accurately"
- Move to 30cm distance (easier)
- Or try 8mm/12mm lens if available

### "Corners not detected during calibration"
- Verify printed square size with ruler
- Ensure pattern is flat (use cardboard backing)
- Check lighting (bright, uniform, no shadows)
- Pattern might be wrong size for your lens

## Need More Help?

See `docs/CALIBRATION_PATTERN_GUIDE.md` for detailed instructions and troubleshooting.
