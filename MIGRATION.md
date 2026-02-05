# Migration Guide: PyuEye to IDS Peak SDK

This guide helps users migrate from the legacy PyuEye SDK to the modern IDS Peak SDK.

## Why Migrate?

### Benefits of IDS Peak SDK:
- ✅ **Modern GenICam standard** - Industry-standard camera interface
- ✅ **Better performance** - Optimized buffer management and lower latency
- ✅ **Serial number identification** - Reliable camera identification
- ✅ **Active development** - Continued support and updates from IDS
- ✅ **Cross-platform** - Better Linux support
- ✅ **Future-proof** - PyuEye is legacy, Peak is the future

### PyuEye Limitations:
- ❌ Legacy SDK with limited updates
- ❌ Device index-only identification (unreliable)
- ❌ Platform-specific quirks
- ❌ Limited GenICam feature support

## Quick Migration Checklist

- [ ] Install IDS Peak SDK (see README.md)
- [ ] Update configuration files
- [ ] Re-run camera detection
- [ ] Update serial numbers in config
- [ ] No code changes needed (scripts use new interface automatically)
- [ ] Re-calibrate cameras (recommended, not required)

## Key Differences

### 1. SDK Installation

**PyuEye:**
```bash
pip install pyueye
```

**IDS Peak:**
```bash
# Install system package first
sudo dpkg -i ids-peak_*.deb

# Then Python bindings
pip install /opt/ids/peak/lib/python*/ids_peak-*.whl
pip install /opt/ids/peak/lib/python*/ids_peak_ipl-*.whl
```

### 2. Exposure Units

**⚠️ IMPORTANT: This is the most common migration issue!**

**PyuEye:** Milliseconds (ms)
```yaml
exposure: 10  # 10 milliseconds
```

**IDS Peak:** Microseconds (µs)
```yaml
exposure_us: 10000  # 10000 microseconds = 10 milliseconds
```

**Conversion:**
```
PyuEye_ms × 1000 = Peak_µs
10 ms × 1000 = 10000 µs
```

### 3. Camera Identification

**PyuEye:** Device index only
```yaml
left_camera:
  device_id: 0

right_camera:
  device_id: 1
```

**IDS Peak:** Serial number (preferred) + device index (fallback)
```yaml
use_serial_numbers: true

left_camera:
  serial_number: "4103123456"  # From list_cameras.py
  device_index: 0               # Fallback

right_camera:
  serial_number: "4103654321"
  device_index: 1
```

### 4. Configuration File Changes

**OLD (PyuEye):**
```yaml
cameras:
  resolution:
    width: 2592
    height: 1944
  
  framerate: 30
  exposure: 10  # milliseconds
  gain: 1.0
  
  left_camera:
    device_id: 0
  
  right_camera:
    device_id: 1

calibration:
  checkerboard:
    rows: 9
    cols: 6
    square_size: 25  # mm
  
  capture:
    num_images: 20
    delay: 1.0
```

**NEW (IDS Peak):**
```yaml
cameras:
  use_serial_numbers: true
  
  left_camera:
    serial_number: "4103123456"
    device_index: 0
    name: "Left Camera"
  
  right_camera:
    serial_number: "4103654321"
    device_index: 1
    name: "Right Camera"
  
  resolution:
    width: 2592
    height: 1944
  
  framerate: 30
  exposure_us: 10000  # microseconds!
  gain: 1.0
  pixel_format: "BGR8"

calibration:
  checkerboard:
    rows: 9
    cols: 6
    square_size_mm: 25.0  # Note: square_size_mm
  
  num_images: 20
  delay_seconds: 1.0
```

### 5. Camera Detection

**PyuEye:**
```bash
# Manual checking of device indices
ls /dev/video*
```

**IDS Peak:**
```bash
# Automatic detection with serial numbers
python scripts/list_cameras.py
```

Output includes serial numbers for config:
```
Camera 0:
  Model:      UI-3680CP Rev.1.2
  Serial:     4103123456  # Use this in config!
  Interface:  U3V
  Status:     ✓ Available
```

## Step-by-Step Migration

### Step 1: Backup Your Configuration

```bash
# Save your old config
cp config/camera_config.yaml config/camera_config_pyueye_backup.yaml
```

### Step 2: Install IDS Peak

See detailed instructions in README.md, section "Installation".

**Linux:**
```bash
sudo dpkg -i ids-peak_*.deb
pip install /opt/ids/peak/lib/python*/ids_peak-*.whl
pip install /opt/ids/peak/lib/python*/ids_peak_ipl-*.whl
```

**Windows:**
Run IDS Peak installer from IDS website.

### Step 3: Detect Your Cameras

```bash
python scripts/list_cameras.py
```

Note the serial numbers displayed for each camera.

### Step 4: Update Configuration

Edit `config/camera_config.yaml`:

1. Add `use_serial_numbers: true`
2. Add `serial_number` fields with values from list_cameras.py
3. Convert exposure from milliseconds to microseconds (multiply by 1000)
4. Change `square_size` to `square_size_mm` (same value)
5. Restructure config as shown in "Configuration File Changes" above

**Quick conversion:**
- `exposure: 10` → `exposure_us: 10000`
- `exposure: 5` → `exposure_us: 5000`
- `exposure: 20` → `exposure_us: 20000`

### Step 5: Test Cameras

```bash
python scripts/test_cameras.py
```

If you get errors:
- Check IDS Peak is installed: `which ids-peak-cockpit`
- Verify serial numbers match: `python scripts/list_cameras.py`
- Check exposure units (microseconds, not milliseconds!)

### Step 6: Update Calibration (Optional)

Your existing calibration files should work, but for best results:

```bash
# Capture new calibration images
python scripts/capture_calibration_images.py

# Re-calibrate
python calibration/calibrate_single_camera.py --camera left
python calibration/calibrate_single_camera.py --camera right
python calibration/calibrate_stereo.py
```

### Step 7: Run Stereo Vision

```bash
python scripts/run_stereo_vision.py
```

Everything should work as before!

## Common Migration Issues

### Issue 1: Cameras Not Detected

**Error:**
```
❌ No IDS Peak cameras found!
```

**Solution:**
1. Verify cameras work in IDS Peak Cockpit
2. Check USB 3.0 connections
3. Linux: Add user to video group
   ```bash
   sudo usermod -a -G video $USER
   # Log out and back in
   ```

### Issue 2: Wrong Exposure

**Symptom:** Images too dark or too bright

**Cause:** Using milliseconds instead of microseconds

**Solution:**
```yaml
# WRONG:
exposure_us: 10  # This is only 0.01ms - way too short!

# CORRECT:
exposure_us: 10000  # This is 10ms
```

Remember: **IDS Peak uses microseconds**
- 1000 µs = 1 ms
- 10000 µs = 10 ms

### Issue 3: Serial Number Not Found

**Error:**
```
Camera with serial 4103123456 not found
Falling back to device index
```

**Solution:**
1. Verify serial number is correct: `python scripts/list_cameras.py`
2. Check for typos in config
3. Ensure `use_serial_numbers: true` is set

### Issue 4: Import Error

**Error:**
```
ImportError: No module named 'ids_peak'
```

**Solution:**
```bash
# Find and install Python bindings
pip install /opt/ids/peak/lib/python*/ids_peak-*.whl
pip install /opt/ids/peak/lib/python*/ids_peak_ipl-*.whl
```

### Issue 5: Performance Difference

**Symptom:** FPS different from PyuEye

**Explanation:** IDS Peak may have different default buffer settings

**Solution:**
- Peak is usually faster with proper configuration
- Adjust resolution if needed
- Check USB 3.0 controllers
- Verify both cameras on separate USB controllers for best performance

## Code Changes (For Custom Scripts)

If you have custom scripts using the old camera interface:

**OLD:**
```python
from src.camera_interface import create_stereo_camera, StereoCamera

stereo_camera = create_stereo_camera(config)
stereo_camera.open()
left, right = stereo_camera.capture_frames()
stereo_camera.close()
```

**NEW:**
```python
from src.camera_interface_peak import create_stereo_camera_from_config

stereo_camera = create_stereo_camera_from_config(config)
stereo_camera.initialize(width, height, exposure_us, gain, pixel_format)
left, right = stereo_camera.capture_stereo_pair()
stereo_camera.release()
```

### Method Name Changes:
- `open()` → `initialize(width, height, exposure_us, gain, pixel_format)`
- `capture_frames()` → `capture_stereo_pair()`
- `close()` → `release()`

## Testing Your Migration

### Quick Test Checklist:

```bash
# 1. Detect cameras
python scripts/list_cameras.py
# ✓ Should show 2 cameras with serial numbers

# 2. Test connections
python scripts/test_cameras.py
# ✓ Should display live feeds from both cameras
# ✓ Should show FPS > 20

# 3. Capture test images
# Press 's' in test_cameras.py
# ✓ Should save test_left_*.png and test_right_*.png

# 4. Check image quality
# Open saved images
# ✓ Proper exposure (not too dark/bright)
# ✓ Sharp focus
# ✓ Correct resolution

# 5. Run full system (if calibrated)
python scripts/run_stereo_vision.py
# ✓ Should show depth map
# ✓ Should show distance measurements
```

## Reverting to PyuEye (If Needed)

If you need to go back to PyuEye:

```bash
# Restore old config
cp config/camera_config_pyueye_backup.yaml config/camera_config.yaml

# Reinstall PyuEye
pip install pyueye

# Update scripts to use old interface
# (Edit scripts to import from camera_interface instead of camera_interface_peak)
```

**Note:** The old `camera_interface.py` is still present for compatibility.

## FAQ

**Q: Do I need to re-calibrate after migration?**
A: Not required, but recommended for best accuracy. Calibration is camera-specific, not SDK-specific.

**Q: Can I use both PyuEye and Peak SDK?**
A: Not simultaneously. Choose one. Peak is recommended for new projects.

**Q: What if list_cameras.py shows 0 cameras?**
A: Camera is either in use by PyuEye/other app, or Peak not installed correctly. Close all camera apps and retry.

**Q: Why is my exposure setting ignored?**
A: Check you're using microseconds (µs), not milliseconds (ms). Multiply your old value by 1000.

**Q: Will my calibration files work?**
A: Yes! Calibration data is stored as numpy arrays and is SDK-independent.

**Q: Can I mix serial number and device index?**
A: Yes. Set `use_serial_numbers: true` and specify both. Peak will try serial first, fall back to index.

## Getting Help

If you encounter issues:

1. Check this migration guide
2. Review README.md troubleshooting section
3. Verify configuration file format
4. Test with IDS Peak Cockpit first
5. Open a GitHub issue with:
   - IDS Peak version
   - Python version
   - Operating system
   - Error messages
   - Config file (sanitized)

## Summary

Migration is straightforward:
1. ✅ Install IDS Peak SDK
2. ✅ Update config with serial numbers
3. ✅ Convert exposure milliseconds → microseconds
4. ✅ Test with list_cameras.py and test_cameras.py
5. ✅ Optionally re-calibrate
6. ✅ Run stereo vision system

**Most important: Remember to multiply exposure by 1000!**

Good luck with your migration! 🚀
