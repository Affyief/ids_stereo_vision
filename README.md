# IDS Stereo Vision System

A complete stereo vision system for IDS u3-3680xcp-c cameras that performs stereo calibration, rectification, depth estimation, and real-time visualization with distance measurements.

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Calibration Guide](#calibration-guide)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)
- [Performance Considerations](#performance-considerations)
- [Future Improvements](#future-improvements)

## Overview

This project implements a complete stereo vision pipeline using two IDS u3-3680xcp-c USB 3.0 cameras. The system provides:

- **Camera Interface**: SDK wrapper for IDS uEye cameras with synchronized capture
- **Stereo Calibration**: Intrinsic and extrinsic camera calibration using chessboard patterns
- **Stereo Rectification**: Image rectification for epipolar geometry alignment
- **Depth Estimation**: Real-time disparity and depth map computation
- **Visualization**: Interactive display with distance measurements and field of view overlay
- **3D Point Cloud**: Export capability for 3D reconstruction

## Hardware Requirements

### Cameras
- **2x IDS uEye U3-3680XCP-C** cameras
  - Sensor: ON Semiconductor AR0521 CMOS
  - Resolution: 2592 x 1944 pixels (5.04 MP)
  - Frame rate: Up to 48 fps at full resolution
  - Interface: USB 3.0
  - Lens mount: C-mount

### Computer
- USB 3.0 ports (2 available)
- Recommended: 8GB+ RAM
- Intel i5 or better processor
- Optional: NVIDIA GPU for CUDA acceleration

### Mounting
- Stereo camera bracket with fixed baseline (60-150mm recommended)
- Cameras should be mounted looking at the same scene with parallel optical axes

## Software Requirements

- Python 3.8 or higher
- IDS uEye SDK 4.95+ (pyueye)
- OpenCV 4.8.0+ with contrib modules
- NumPy, PyYAML, SciPy, Matplotlib
- Optional: Open3D for 3D visualization

## Installation

### 1. Install IDS uEye SDK

Download and install the IDS Software Suite from:
https://en.ids-imaging.com/downloads.html

The SDK includes drivers and the uEye Cockpit application for camera configuration.

### 2. Clone Repository

```bash
git clone https://github.com/Affyief/ids_stereo_vision.git
cd ids_stereo_vision
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Camera Connection

```bash
python tests/test_camera_connection.py
```

This will test camera detection, frame capture, synchronization, and resolution.

## Project Structure

```
ids_stereo_vision/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── config/
│   ├── camera_params.yaml            # Calibration results (generated)
│   └── stereo_config.yaml            # System configuration
├── calibration/
│   ├── __init__.py
│   ├── calibrate_cameras.py          # Calibration algorithms
│   └── chessboard_images/            # Calibration image storage
│       ├── left/                     # Left camera images
│       └── right/                    # Right camera images
├── src/
│   ├── __init__.py
│   ├── camera_interface.py           # IDS camera SDK wrapper
│   ├── stereo_processor.py           # Stereo matching and depth
│   ├── visualization.py              # Display and visualization
│   └── utils.py                      # Helper functions
├── scripts/
│   ├── capture_calibration_images.py # Capture calibration pairs
│   ├── run_calibration.py            # Run calibration process
│   ├── run_stereo_system.py          # Main application
│   ├── calculate_calibration_pattern.py # Pattern calculator
│   └── generate_pattern.py           # Pattern generator
├── docs/
│   └── CALIBRATION_PATTERN_GUIDE.md  # Detailed pattern guide
├── tests/
│   └── test_camera_connection.py     # Camera connection tests
└── output/                           # Saved frames and data (created)
```

## Quick Start

### 1. Configure System

Edit `config/stereo_config.yaml` to match your setup:

```yaml
cameras:
  left:
    device_id: 0        # First camera
    exposure: 10        # Exposure in ms
    gain: 1.0          # Hardware gain
  right:
    device_id: 1        # Second camera
    exposure: 10
    gain: 1.0
```

### 2. Calibrate Cameras

#### Step 2a: Determine Optimal Pattern for Your Setup

**IMPORTANT**: Pattern size depends on your lens focal length and calibration distance!

**For 20cm calibration distance:**

1. **Find your lens focal length** (check the lens marking - e.g., "6mm", "8mm")

2. **Calculate the optimal pattern:**
   ```bash
   python scripts/calculate_calibration_pattern.py --focal 6 --distance 200
   ```
   This will recommend the best pattern for your specific setup.

3. **Generate the pattern:**
   ```bash
   # Use the values from the calculator output
   python scripts/generate_pattern.py --rows 8 --cols 11 --size 12
   ```

**Quick Start Examples:**
- **6mm lens at 20cm**: Use `config/stereo_config_6mm_20cm.yaml`
- **8mm lens at 20cm**: Use `config/stereo_config_8mm_20cm.yaml`
- **6mm lens at 30cm** (easier): Use `config/stereo_config_6mm_30cm.yaml`

See `docs/CALIBRATION_PATTERN_GUIDE.md` for detailed guidance.

#### Step 2b: Print Calibration Pattern

**Critical**: Print at 100% scale (no "fit to page")!

1. Open the generated `calibration_pattern.png`
2. Print settings: 100% scale, highest quality
3. Verify with ruler: squares must be exact size
4. Mount on flat, rigid surface

#### Step 2c: Capture Calibration Images

```bash
python scripts/capture_calibration_images.py --count 30
```

- Position the chessboard at different angles and distances
- Press SPACE to capture each image pair
- Capture at least 20-30 pairs for good calibration
- Cover the entire field of view

#### Step 2d: Run Calibration

```bash
python scripts/run_calibration.py --visualize
```

This will:
- Detect chessboard corners in all image pairs
- Compute intrinsic parameters for each camera
- Compute extrinsic parameters (rotation and translation)
- Generate rectification transforms
- Save results to `config/camera_params.yaml`

### 3. Run Stereo System

```bash
python scripts/run_stereo_system.py
```

## Calibration Guide

### Pattern Requirements

The default calibration uses a chessboard pattern with:
- **9x6 internal corners** (10x7 squares)
- **25mm square size**

You can modify these in `config/stereo_config.yaml`:

```yaml
system:
  calibration_pattern:
    type: "chessboard"
    width: 9              # Internal corners width
    height: 6             # Internal corners height
    square_size: 25.0     # Square size in mm
```

### Best Practices

1. **Pattern Quality**
   - Print on flat, rigid material
   - Ensure squares are perfectly square
   - Good contrast between black and white

2. **Image Capture**
   - Capture from various angles (tilted, rotated)
   - Capture at various distances (near, medium, far)
   - Cover all areas of the image
   - Ensure pattern is fully visible in both cameras
   - Avoid motion blur

3. **Lighting**
   - Use uniform, diffuse lighting
   - Avoid shadows and reflections
   - Consistent exposure across all images

4. **Expected Results**
   - Calibration error < 0.5 pixels is excellent
   - Calibration error < 1.0 pixels is good
   - Error > 1.5 pixels suggests issues with captured images

### Validation

After calibration, check the results:

```yaml
stereo:
  baseline: 120.5           # Distance between cameras (mm)
  calibration_error: 0.42   # RMS error (pixels)
```

- **Baseline** should match your physical camera spacing
- **Calibration error** should be < 1.0 pixels

## Usage

### Main Application Controls

When running `run_stereo_system.py`:

- **Q**: Quit application
- **S**: Save current frame and depth map
- **M**: Toggle interactive measurement mode
- **P**: Save 3D point cloud (PLY format)
- **D**: Toggle disparity map display
- **R**: Toggle raw view (no overlays)

### Display Windows

The main window shows four views:
1. **Top-Left**: Rectified image with distance overlays
2. **Top-Right**: Depth map (color-coded)
3. **Bottom-Left**: Raw rectified image
4. **Bottom-Right**: Disparity map

### Interactive Measurement

With measurement mode (press M):
- Move mouse over the image
- Distance at cursor position is displayed
- Useful for spot measurements

### Saving Data

Press **S** to save:
- `left_<timestamp>.png`: Rectified left image
- `depth_<timestamp>.npy`: Raw depth data (NumPy array)
- `depth_vis_<timestamp>.png`: Colored depth visualization

Press **P** to save:
- `pointcloud_<timestamp>.ply`: 3D point cloud

View PLY files with tools like:
- CloudCompare
- MeshLab
- Open3D viewer

## Configuration

### System Configuration (`config/stereo_config.yaml`)

#### Stereo Matching Parameters

```yaml
stereo_matching:
  algorithm: "SGBM"         # "SGBM" or "BM"
  min_disparity: 0          # Minimum disparity
  num_disparities: 128      # Must be divisible by 16
  block_size: 11            # Matching block size (odd)
  uniqueness_ratio: 10      # Uniqueness threshold
  speckle_window_size: 100  # Speckle filter window
  speckle_range: 32         # Speckle filter range
  disp12_max_diff: 1        # Left-right consistency
```

**Algorithm Choice:**
- **SGBM** (Semi-Global Block Matching): Better quality, slower
- **BM** (Block Matching): Faster, suitable for real-time

**Tuning Tips:**
- Increase `num_disparities` for greater depth range
- Decrease `block_size` for finer details (but noisier)
- Increase `uniqueness_ratio` to filter unreliable matches

#### Display Parameters

```yaml
display:
  window_width: 1920        # Display window width
  window_height: 1080       # Display window height
  fps_target: 30            # Target frame rate
  depth_color_map: "JET"    # Colormap: JET, HOT, RAINBOW, etc.
  min_distance: 100         # Minimum depth (mm)
  max_distance: 3000        # Maximum depth (mm)
```

### Camera Parameters (`config/camera_params.yaml`)

Generated by calibration. Contains:
- Intrinsic matrices and distortion coefficients
- Rotation and translation between cameras
- Rectification transforms
- Disparity-to-depth mapping matrix

## API Documentation

### Camera Interface

```python
from src.camera_interface import StereoCameraSystem

# Initialize stereo system
stereo = StereoCameraSystem(left_id=0, right_id=1)
stereo.initialize(width=2592, height=1944, exposure=10, gain=1)

# Capture frames
left_frame, right_frame = stereo.capture_stereo_pair()

# Release resources
stereo.release()
```

### Stereo Processing

```python
from src.stereo_processor import StereoProcessor

# Initialize processor
processor = StereoProcessor(calibration_params, config)

# Process stereo pair
results = processor.process_stereo_pair(left_frame, right_frame)
# Returns: rectified_left, rectified_right, disparity, depth

# Compute 3D points
points_3d, colors = processor.compute_point_cloud(disparity, left_frame)
```

### Visualization

```python
from src.visualization import StereoVisualizer

# Initialize visualizer
viz = StereoVisualizer(config)

# Display results
key = viz.display_results(rectified_left, depth, disparity)

# Interactive measurement
viz.toggle_measurement_mode()
viz.setup_mouse_callback()
```

## Troubleshooting

### Camera Connection Issues

**Problem**: Cameras not detected

**Solutions**:
- Check USB 3.0 connection (use blue USB ports)
- Install IDS uEye SDK drivers
- Use IDS uEye Cockpit to verify cameras are detected
- Check device IDs in configuration
- Try different USB ports
- Restart computer after driver installation

### Calibration Problems

**Problem**: Corners not detected

**Solutions**:
- Ensure good lighting
- Verify pattern is flat and correctly printed
- Check pattern dimensions in config match physical pattern
- Adjust camera exposure if image is too bright/dark
- Ensure entire pattern is visible in both cameras

**Problem**: High calibration error

**Solutions**:
- Capture more images (30+ recommended)
- Improve image quality (lighting, focus, no motion blur)
- Cover more angles and positions
- Use larger chessboard squares if pattern is too small
- Ensure cameras are firmly mounted

### Depth Estimation Issues

**Problem**: Noisy depth map

**Solutions**:
- Increase `block_size` in stereo_matching config
- Increase `speckle_window_size`
- Use SGBM instead of BM algorithm
- Improve lighting (uniform, no shadows)
- Adjust exposure for consistent brightness

**Problem**: Limited depth range

**Solutions**:
- Increase `num_disparities` (must be multiple of 16)
- Verify baseline is correctly calibrated
- Ensure objects are within working distance

### Performance Issues

**Problem**: Low frame rate

**Solutions**:
- Reduce image resolution in camera initialization
- Use BM algorithm instead of SGBM
- Decrease `num_disparities`
- Decrease `block_size`
- Use GPU acceleration if available
- Close other applications

## Performance Considerations

### Resolution vs Frame Rate

| Resolution | Expected FPS | Use Case |
|------------|-------------|----------|
| 2592x1944 | 15-20 fps | High-quality depth |
| 1296x972 | 30-40 fps | Real-time applications |
| 648x486 | 60+ fps | Fast tracking |

### Depth Range vs Accuracy

The depth calculation formula:
```
depth (mm) = (baseline * focal_length) / disparity
```

- **Larger baseline**: Greater depth range, but reduced overlap
- **Smaller baseline**: Better close-range accuracy
- **Typical baseline**: 80-120mm for 0.5-5m working distance

### USB Bandwidth

Two cameras at full resolution require ~400 MB/s bandwidth:
- Ensure both cameras are on separate USB 3.0 controllers if possible
- Reduce resolution if bandwidth issues occur
- Use USB 3.1 Gen 2 for best performance

## Future Improvements

- [ ] GPU acceleration using CUDA
- [ ] Hardware triggering for better synchronization
- [ ] Temporal filtering for smoother depth maps
- [ ] Automatic camera discovery and configuration
- [ ] Web interface for remote monitoring
- [ ] ROS integration for robotics applications
- [ ] Real-time object detection and tracking
- [ ] Automatic exposure and gain control
- [ ] Multi-camera support (>2 cameras)
- [ ] Depth map post-processing filters

## Examples and Expected Output

### Calibration Output

```
Calibrating stereo system with 25 valid pairs...
Stereo calibration successful. RMS error: 0.4231
Baseline: 120.45 mm
==================================================
CALIBRATION SUMMARY
==================================================
Baseline: 120.45 mm
Calibration error: 0.4231 pixels
Left camera focal length: fx=1548.32, fy=1549.87
Right camera focal length: fx=1547.98, fy=1549.32
==================================================
```

### Runtime Performance

```
Camera initialized: 2592x1944
Stereo processor initialized
Average FPS: 18.5
Depth range: 300-2500mm
```

## References

- IDS uEye SDK Documentation: https://en.ids-imaging.com/manuals.html
- OpenCV Stereo Vision: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
- Camera Calibration Tutorial: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- IDS u3-3680xcp-c Specifications: https://www.edmundoptics.com/p/ids-imaging-u3-3680xcp-c-hq-118-usb-color-camera-rev-12/55756/

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For issues related to:
- **IDS cameras**: Contact IDS Imaging support
- **This software**: Open a GitHub issue
- **OpenCV**: Refer to OpenCV documentation

## Acknowledgments

- IDS Imaging for camera hardware and SDK
- OpenCV community for computer vision algorithms
- Python open-source community