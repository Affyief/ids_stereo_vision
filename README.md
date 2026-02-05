# IDS Peak Stereo Vision System

A complete, production-ready stereo vision system for IDS U3-3680XCP-C cameras using the modern **IDS Peak SDK** (GenICam/GenTL interface). Features real-time depth computation, interactive distance measurements, and comprehensive calibration tools.

## 🎯 Key Features

- **Modern IDS Peak SDK**: Uses GenICam interface (NOT legacy PyuEye)
- **Serial Number Identification**: Reliable camera identification in multi-camera setups
- **Real-time Depth Mapping**: Live stereo matching with SGBM algorithm
- **Interactive Visualization**: Click-to-measure distances, color-coded depth maps
- **Complete Calibration Pipeline**: Automated tools for intrinsic and extrinsic calibration
- **Post-processing**: WLS filtering for high-quality depth maps
- **Production Ready**: Clean, modern Python code with proper error handling

## 📷 Hardware Requirements

### Cameras
**IDS uEye U3-3680XCP-C-HQ:**
- Sensor: ON Semiconductor AR0521 CMOS
- Resolution: 2592 × 1944 pixels (5.04 MP)
- Sensor size: 5.702 × 4.277 mm (1/2.5" format)
- Pixel size: 2.2 × 2.2 µm
- Frame rate: Up to 48-49 fps at full resolution
- Interface: USB 3.0 (USB3 Vision, GenICam compliant)
- Lens mount: C-mount

### Additional Hardware
- 2× C-mount lenses (6mm, 8mm, or 12mm recommended)
- Rigid stereo mounting bracket
- USB 3.0 host controllers (one per camera recommended for best performance)
- Checkerboard calibration pattern (9×6 with 25mm squares recommended)

### Computer Requirements
- USB 3.0 ports (2 available)
- Linux (Ubuntu 20.04+) or Windows 10/11
- Python 3.7+
- 8GB+ RAM recommended

## 💿 Installation

### 1. Install IDS Peak SDK

#### Linux (Ubuntu/Debian)

```bash
# Download IDS peak from https://en.ids-imaging.com/downloads.html
# Select: IDS peak -> Linux -> IDS peak 2.x

# Install the package
sudo dpkg -i ids-peak_*.deb

# Verify installation
/opt/ids/peak/bin/ids-peak-cockpit

# Set up camera permissions (required for non-root access)
sudo usermod -a -G video $USER
# Log out and back in for permissions to take effect

# Install Python bindings
pip install /opt/ids/peak/lib/python*/ids_peak-*.whl
pip install /opt/ids/peak/lib/python*/ids_peak_ipl-*.whl
```

**Note**: The Python bindings path may vary. Check your installation directory:
```bash
ls /opt/ids/peak/lib/
```

#### Windows

1. Download IDS peak installer from [https://en.ids-imaging.com/downloads.html](https://en.ids-imaging.com/downloads.html)
2. Run the installer (administrator rights required)
3. Follow installation wizard
4. Python bindings are usually installed automatically
5. Verify installation by opening IDS peak Cockpit

If Python bindings not installed:
```powershell
# Find IDS peak installation directory (usually C:\Program Files\IDS\peak)
pip install "C:\Program Files\IDS\peak\lib\python*\ids_peak-*.whl"
pip install "C:\Program Files\IDS\peak\lib\python*\ids_peak_ipl-*.whl"
```

### 2. Clone Repository and Install Dependencies

```bash
# Clone repository
git clone https://github.com/Affyief/ids_stereo_vision.git
cd ids_stereo_vision

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
# List detected cameras
python scripts/list_cameras.py

# Test camera connections
python scripts/test_cameras.py
```

If cameras are detected, you're ready to proceed! 🎉

## 🚀 Quick Start

### Step 1: Detect Cameras

```bash
python scripts/list_cameras.py
```

This will show all connected IDS Peak cameras with their serial numbers. Example output:
```
✓ Found 2 IDS Peak camera(s):

Camera 0:
  Model:      UI-3680CP Rev.1.2
  Serial:     4103123456
  Interface:  U3V
  Status:     ✓ Available
  Test:       ✓ Capture successful

Camera 1:
  Model:      UI-3680CP Rev.1.2
  Serial:     4103654321
  Interface:  U3V
  Status:     ✓ Available
  Test:       ✓ Capture successful
```

### Step 2: Configure Cameras

Edit `config/camera_config.yaml` with your camera serial numbers:

```yaml
cameras:
  use_serial_numbers: true
  
  left_camera:
    serial_number: "4103123456"  # Copy from list_cameras.py output
    device_index: 0
  
  right_camera:
    serial_number: "4103654321"  # Copy from list_cameras.py output
    device_index: 1
  
  resolution:
    width: 2592
    height: 1944
  
  exposure_us: 10000  # 10ms (Peak uses microseconds!)
  gain: 1.0
```

**Important**: IDS Peak uses **microseconds** for exposure (not milliseconds like PyuEye).

### Step 3: Measure Baseline

Measure the distance between the optical centers of your two cameras in millimeters. Update `config/stereo_config.yaml`:

```yaml
stereo:
  baseline_mm: 65.0  # CHANGE THIS to your actual measurement
```

This is critical for accurate depth measurements!

### Step 4: Test Cameras

```bash
python scripts/test_cameras.py
```

This verifies both cameras can capture synchronized frames. Press 'q' to quit, 's' to save test images.

### Step 5: Capture Calibration Images

```bash
python scripts/capture_calibration_images.py
```

**Calibration Instructions:**
1. Print a checkerboard pattern (9×6 internal corners, 25mm squares)
2. Mount it on a flat, rigid surface
3. Position it in view of both cameras
4. Press **SPACE** to capture when checkerboard is detected
5. Move checkerboard to different positions, angles, and distances
6. Capture **at least 20 pairs** (30+ recommended)
7. Cover entire field of view including corners
8. Press 'q' when done

**Tips:**
- Ensure good lighting
- Keep checkerboard sharp (no motion blur)
- Vary distance: near (30cm), medium (1m), far (2m+)
- Vary angles: tilted left/right, up/down
- Cover all areas of the frame

### Step 6: Calibrate Cameras

```bash
# Calibrate left camera
python calibration/calibrate_single_camera.py --camera left

# Calibrate right camera
python calibration/calibrate_single_camera.py --camera right

# Perform stereo calibration
python calibration/calibrate_stereo.py
```

**Good calibration**: Reprojection error < 1.0 pixels

If error is high:
- Capture more images (30-40 pairs)
- Ensure better coverage of field of view
- Check focus and lighting
- Verify checkerboard dimensions in config

### Step 7: Run Stereo Vision System

```bash
python scripts/run_stereo_vision.py
```

The application will display:
- Left and right rectified images
- Color-coded depth map
- Distance measurements grid
- FPS and statistics

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save current frame and depth map |
| `c` | Cycle through colormaps (TURBO, JET, HSV, etc.) |
| `f` | Toggle WLS filtering |
| `m` | Toggle measurement grid |
| `x` | Toggle crosshair |
| `+`/`-` | Adjust number of disparities |
| `[`/`]` | Adjust block size |
| `p` | Pause/resume |
| `h` | Show help |

## 📁 Project Structure

```
ids_stereo_vision/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore
│
├── config/                            # Configuration files
│   ├── camera_config.yaml            # Camera settings, serial numbers
│   └── stereo_config.yaml            # Stereo processing parameters
│
├── calibration/                       # Calibration scripts
│   ├── calibrate_single_camera.py    # Single camera calibration
│   ├── calibrate_stereo.py           # Stereo calibration
│   └── calibration_images/           # Captured calibration images
│       ├── left/                     # Left camera images
│       └── right/                    # Right camera images
│
├── calibration_data/                  # Saved calibration data
│   ├── left_camera_calibration.npz   # Left camera parameters
│   ├── right_camera_calibration.npz  # Right camera parameters
│   └── stereo_calibration.npz        # Stereo parameters (R, T, Q)
│
├── src/                               # Core source code
│   ├── __init__.py
│   ├── camera_interface_peak.py      # IDS Peak SDK interface
│   ├── camera_interface.py           # Legacy interface (deprecated)
│   ├── stereo_processor.py           # Stereo matching and depth
│   ├── depth_visualizer.py           # Visualization and overlays
│   └── utils.py                      # Utility functions
│
└── scripts/                           # Executable scripts
    ├── list_cameras.py               # List detected cameras
    ├── test_cameras.py               # Test camera connections
    ├── capture_calibration_images.py # Capture calibration images
    └── run_stereo_vision.py          # Main application
```

## ⚙️ Configuration

### Camera Configuration (`config/camera_config.yaml`)

```yaml
cameras:
  # Serial number identification (recommended)
  use_serial_numbers: true
  
  left_camera:
    serial_number: "4103123456"  # From list_cameras.py
    device_index: 0               # Fallback
    name: "Left Camera"
  
  right_camera:
    serial_number: "4103654321"
    device_index: 1
    name: "Right Camera"
  
  # Image settings
  resolution:
    width: 2592   # Full resolution
    height: 1944
    # For better performance:
    # width: 1296
    # height: 972
  
  framerate: 30
  
  # Exposure in microseconds (IDS Peak uses µs!)
  exposure_us: 10000  # 10ms
  
  # Gain (0.0 to max, typically 0-24 dB)
  gain: 1.0
  
  # Pixel format: BGR8, RGB8, Mono8
  pixel_format: "BGR8"

calibration:
  checkerboard:
    rows: 9           # Internal corners
    cols: 6
    square_size_mm: 25.0
  
  num_images: 20
  delay_seconds: 1.0
```

### Stereo Configuration (`config/stereo_config.yaml`)

```yaml
stereo:
  # CRITICAL: Measure and update this!
  baseline_mm: 65.0  # Distance between camera optical centers
  
  # Algorithm: "BM" (fast) or "SGBM" (better quality)
  algorithm: "SGBM"
  
  # StereoSGBM parameters
  sgbm:
    min_disparity: 0
    num_disparities: 128    # Must be divisible by 16
    block_size: 11          # Odd number, 5-21
    P1: 600                 # Smoothness penalty
    P2: 2400                # Smoothness penalty
    disp12_max_diff: 1
    uniqueness_ratio: 10
    speckle_window_size: 100
    speckle_range: 32
    mode: "SGBM_MODE_HH"
  
  # Post-processing
  post_process:
    use_wls_filter: true    # Better quality, ~20% slower
    wls_lambda: 8000
    wls_sigma: 1.5
    median_blur_ksize: 5
  
  # Visualization
  visualization:
    colormap: "TURBO"       # TURBO, JET, HSV, VIRIDIS, etc.
    min_distance_m: 0.3
    max_distance_m: 10.0
    measurement_points: 9   # Grid size
```

## 🔧 Troubleshooting

### Cameras Not Detected

**Problem**: `list_cameras.py` shows no cameras

**Solutions**:
1. Check USB 3.0 connections (must be USB 3.0, not 2.0)
2. Verify cameras work in IDS Peak Cockpit
3. Linux: Check permissions
   ```bash
   sudo usermod -a -G video $USER
   # Log out and back in
   ```
4. Try different USB ports/controllers
5. Ensure no other application is using cameras

### IDS Peak SDK Not Found

**Problem**: `ImportError: No module named 'ids_peak'`

**Solutions**:
1. Verify IDS Peak installation:
   ```bash
   # Linux:
   ls /opt/ids/peak/lib/python*/
   
   # Windows:
   dir "C:\Program Files\IDS\peak\lib\python*"
   ```

2. Install Python bindings:
   ```bash
   pip install /opt/ids/peak/lib/python*/ids_peak-*.whl
   pip install /opt/ids/peak/lib/python*/ids_peak_ipl-*.whl
   ```

3. Check Python version matches bindings (3.7, 3.8, 3.9, etc.)

### Poor Calibration Results

**Problem**: High reprojection error (> 1.0 pixels)

**Solutions**:
1. Capture more images (30-40 pairs)
2. Improve coverage:
   - Cover all areas of field of view
   - Include corners and edges
   - Vary distances and angles
3. Check image quality:
   - Ensure sharp focus
   - Good lighting
   - No motion blur
4. Verify checkerboard dimensions match config
5. Use rigid, flat checkerboard (no warping)

### Low Frame Rate

**Problem**: FPS < 15

**Solutions**:
1. Reduce resolution:
   ```yaml
   resolution:
     width: 1296
     height: 972
   ```

2. Disable WLS filtering:
   ```yaml
   use_wls_filter: false
   ```

3. Reduce disparities:
   ```yaml
   num_disparities: 64  # From 128
   ```

4. Use StereoBM instead of SGBM:
   ```yaml
   algorithm: "BM"
   ```

5. Close other applications
6. Use faster computer

### Inaccurate Depth Measurements

**Problem**: Measured distances are incorrect

**Solutions**:
1. **Verify baseline measurement** - This is the most common issue!
   - Measure center-to-center distance between lenses
   - Update `baseline_mm` in config
   - 1mm error = significant depth error

2. Check calibration quality:
   - Reprojection error should be < 1.0
   - Re-calibrate if needed

3. Ensure cameras are rigidly mounted:
   - No movement between cameras
   - Parallel alignment helps

4. Verify both cameras have similar exposure/brightness

5. Check scene has sufficient texture:
   - Plain white walls don't work well
   - Add texture or patterns

### No Depth in Certain Areas

**Problem**: Black areas in depth map

**Solutions**:
1. Increase disparities for far objects:
   ```yaml
   num_disparities: 160  # From 128
   ```

2. Ensure adequate lighting

3. Add texture to plain surfaces

4. Check both cameras see the area

5. Objects too close or too far:
   - Minimum distance: ~30-50 cm
   - Maximum distance: ~5-10 m (depends on baseline)

## 📊 Performance Tuning

### For Better FPS:
```yaml
resolution:
  width: 1296
  height: 972

algorithm: "BM"  # Instead of SGBM

sgbm:
  num_disparities: 64
  block_size: 15

post_process:
  use_wls_filter: false
```
**Expected**: 30-60 fps

### For Better Quality:
```yaml
resolution:
  width: 2592
  height: 1944

algorithm: "SGBM"

sgbm:
  num_disparities: 128
  block_size: 7

post_process:
  use_wls_filter: true
```
**Expected**: 10-20 fps

### Balanced:
```yaml
resolution:
  width: 1296
  height: 972

algorithm: "SGBM"

sgbm:
  num_disparities: 96
  block_size: 11

post_process:
  use_wls_filter: true
```
**Expected**: 15-25 fps

## 📐 Understanding Stereo Vision

### Depth Calculation

Stereo vision uses triangulation to calculate depth:

```
Z = (f × B) / d

Where:
  Z = distance to object (depth)
  f = focal length in pixels
  B = baseline (distance between cameras)
  d = disparity in pixels
```

**Key Insights**:
- Larger baseline → Better accuracy at long distances
- Smaller baseline → Shorter minimum distance
- Longer focal length → Better depth resolution
- More disparities → Larger depth range

### Typical Distance Ranges

With 65mm baseline, 8mm lens, 1296×972 resolution:

| Distance | Accuracy |
|----------|----------|
| 0.3-0.5m | Poor (minimum range) |
| 0.5-2.0m | Excellent |
| 2.0-5.0m | Good |
| 5.0-10m  | Fair |
| 10m+     | Poor |

### Calibration Theory

**Intrinsic Parameters** (camera internal):
- Camera matrix (K): focal length (fx, fy), principal point (cx, cy)
- Distortion coefficients: lens distortion correction

**Extrinsic Parameters** (camera relationships):
- Rotation matrix (R): 3×3 rotation between cameras
- Translation vector (T): 3D translation (includes baseline)
- Essential matrix (E): combines R and T
- Fundamental matrix (F): relates image points

**Rectification**:
- Aligns image planes so corresponding points have same y-coordinate
- Simplifies stereo matching (search along horizontal lines only)
- Uses R1, R2, P1, P2, Q matrices

## 🔒 Important Notes

### Exposure Units
⚠️ **IDS Peak uses MICROSECONDS** for exposure, not milliseconds!
- 1000 µs = 1 ms
- 10000 µs = 10 ms (typical value)
- Do NOT confuse with PyuEye which used milliseconds

### Serial Numbers vs Device Index
- **Serial numbers** are preferred (reliable, consistent)
- **Device index** works but may change if USB enumeration changes
- Always use serial numbers in production systems

### Camera Synchronization
- Current implementation: software-triggered sequential capture
- For true hardware sync: Use GPIO trigger on both cameras (future enhancement)
- Sequential capture works well for most applications (< 1ms difference)

### Baseline Measurement
- Measure center-to-center distance between lenses
- Use calipers for precision
- Critical for accurate depth - measure carefully!

## 🚧 Future Enhancements

- [ ] Hardware-triggered synchronized capture
- [ ] 3D point cloud generation (PLY export)
- [ ] GPU acceleration (CUDA)
- [ ] ROS integration
- [ ] Automatic exposure/gain control
- [ ] Video recording with depth
- [ ] Web interface for remote viewing
- [ ] Object detection with distance

## 📚 References

- [IDS Peak Documentation](https://www.ids-imaging.com/manuals/ids-peak/)
- [OpenCV Stereo Vision](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- [Camera Calibration Guide](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [GenICam Standard](https://www.emva.org/standards-technology/genicam/)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and research purposes.

## 💬 Support

For issues and questions:
1. Check this README and troubleshooting section
2. Verify configuration files
3. Check calibration quality
4. Open an issue on GitHub with:
   - IDS Peak version
   - Python version
   - Operating system
   - Error messages
   - What you've already tried

## 🙏 Acknowledgments

- OpenCV team for computer vision library
- IDS Imaging for cameras and SDK
- Computer vision community for stereo algorithms

---

**Made with ❤️ for precision depth measurement**

**Happy Stereo Vision! 📷📷📐**
