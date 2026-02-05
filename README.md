# IDS Stereo Vision System

A complete stereo vision system for IDS U3-3680XCP-C cameras that captures synchronized stereo images, computes depth maps, and displays real-time distance measurements.

## Features

- **Dual Camera Support**: Interface with two IDS U3-3680XCP-C cameras via modern IDS Peak SDK
- **Color Camera Support**: Full RGB/BGR color capture with configurable pixel formats
- **Camera Calibration**: Complete calibration pipeline for intrinsic and extrinsic parameters
- **Stereo Matching**: High-quality depth computation using SGBM (Semi-Global Block Matching)
- **Real-time Visualization**: Live depth maps with color-coded distance overlays
- **Interactive Measurements**: Click-to-measure distance at any point in the scene
- **Configurable Parameters**: YAML-based configuration for easy customization
- **Post-processing**: WLS filtering for disparity refinement and noise reduction

## Hardware Requirements

### Cameras
- **Model**: IDS U3-3680XCP-C (or compatible)
- **Sensor**: ON Semiconductor AR0521 CMOS
- **Resolution**: 2592 x 1944 pixels (5.04 MP)
- **Sensor Size**: 5.702 mm x 4.277 mm (1/2.5" format)
- **Pixel Size**: 2.2 µm x 2.2 µm
- **Interface**: USB 3.0 (USB3 Vision, GenICam compliant)
- **Lens Mount**: C-mount
- **Frame Rate**: Up to 48-49 fps at full resolution

### Additional Hardware
- C-mount lenses (6mm, 8mm, or 12mm recommended)
- Sturdy mounting bracket for both cameras
- USB 3.0 host controller (one per camera recommended)
- Checkerboard pattern for calibration (e.g., 9x6 with 25mm squares)

## Lens Configuration

### Supported Lens: IDS-2M12-C0620

This system is configured for the **IDS-2M12-C0620** 6mm C-mount lens:

| Specification | Value |
|--------------|-------|
| **Focal Length** | 6mm |
| **Format** | 1/2" |
| **F-number** | f/2.0 |
| **Mount** | C-mount |
| **Working Distance** | 100mm to ∞ |
| **Distortion** | <3% |

#### Optical Performance

With the IDS U3-3680XCP-C camera (2592×1944, 5.7×4.3mm sensor):

- **Focal Length (pixels):** fx = fy ≈ 2727 pixels
- **Field of View:**
  - Horizontal: 50.7°
  - Vertical: 39.2°
  - Diagonal: 62.4°

#### Configuration

Lens specifications are configured in `config/camera_config.yaml`:

```yaml
cameras:
  lens:
    model: "IDS-2M12-C0620"
    focal_length_mm: 6.0
    f_number: 2.0
```

These parameters are used to:
1. Initialize camera matrix for calibration
2. Calculate field of view
3. Estimate depth measurement accuracy

#### Using Different Lenses

To use a different lens:

1. Update `focal_length_mm` in `camera_config.yaml`
2. Recalibrate cameras (see [Calibration](#calibration))
3. Common focal lengths for IDS lenses: 4mm, 6mm, 8mm, 12mm, 16mm

**Note:** Ensure your lens format (1/2", 1/3", etc.) is compatible with the 1/2.5" sensor.

## Software Requirements

### Operating System
- Linux (Ubuntu 20.04+ recommended)
- Windows 10/11
- macOS

### Dependencies
- Python 3.7+
- OpenCV 4.5+ with contrib modules
- NumPy
- PyYAML
- IDS Peak SDK (latest generation, GenICam/GenTL based)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Affyief/ids_stereo_vision.git
cd ids_stereo_vision
```

### 2. Install IDS Peak SDK

**Linux:**
1. Download IDS Peak from: https://en.ids-imaging.com/downloads.html
2. Install the package:
   ```bash
   sudo dpkg -i ids-peak_*.deb
   sudo apt-get install -f
   ```
3. Install Python bindings:
   ```bash
   # Find the wheel files
   ls /opt/ids/peak/generic_sdk/ipl/binding/python/wheel/x86_64/
   ls /opt/ids/peak/generic_sdk/api/binding/python/wheel/x86_64/
   
   # Install them
   pip install /opt/ids/peak/generic_sdk/ipl/binding/python/wheel/x86_64/ids_peak_ipl-*.whl
   pip install /opt/ids/peak/generic_sdk/api/binding/python/wheel/x86_64/ids_peak-*.whl
   ```

**Windows:**
1. Download and install IDS Peak from IDS website
2. Python bindings location:
   ```
   C:\Program Files\IDS\ids_peak\generic_sdk\ipl\binding\python\wheel\x86_64\
   C:\Program Files\IDS\ids_peak\generic_sdk\api\binding\python\wheel\x86_64\
   ```
3. Install with pip (use correct paths)

### 3. Install Python Dependencies

```bash
cd ids_stereo_vision
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Test Cameras

```bash
python scripts/test_cameras.py
```

## Quick Start

### 1. Test Camera Connections
```bash
python scripts/test_cameras.py
```

This will:
- List all available IDS Peak cameras
- Display camera model, serial number, and interface
- Verify both cameras are accessible
- Display live feeds from both cameras
- Show FPS performance
- Save test images (press 's')

### 2. Capture Calibration Images
```bash
python scripts/capture_calibration_images.py
```

Instructions:
1. Print a checkerboard pattern (default: 9x6, 25mm squares)
2. Mount the checkerboard on a flat surface
3. Position it in front of both cameras
4. Press SPACE to capture images (capture at least 20 pairs)
5. Move checkerboard to different positions and angles
6. Cover the entire field of view

**Tips for Good Calibration:**
- Include images at various depths (near and far)
- Tilt the checkerboard at different angles
- Cover corners and edges of the field of view
- Ensure good lighting and sharp focus
- Verify checkerboard is visible in BOTH cameras

### 3. Calibrate Individual Cameras
```bash
# Calibrate left camera
python calibration/calibrate_single_camera.py --camera left

# Calibrate right camera
python calibration/calibrate_single_camera.py --camera right
```

The script will:
- Process calibration images
- Compute intrinsic parameters (camera matrix, distortion coefficients)
- Report reprojection error
- Save calibration to `calibration_data/`

**Good calibration**: reprojection error < 1.0 pixels

### 4. Perform Stereo Calibration
```bash
python calibration/calibrate_stereo.py
```

This computes:
- Rotation matrix (R) and translation vector (T) between cameras
- Baseline distance
- Rectification parameters
- Projection matrices for 3D reconstruction

### 5. Run Stereo Vision System
```bash
python scripts/run_stereo_vision.py
```

The application will:
- Initialize both cameras
- Load calibration data
- Compute depth maps in real-time
- Display color-coded depth visualization
- Show distance measurements

## Configuration

### Camera Configuration (`config/camera_config.yaml`)

```yaml
cameras:
  # Pixel format for color cameras
  # Options: BGR8 (color, OpenCV native), RGB8 (color), BayerRG8 (raw Bayer), Mono8 (grayscale)
  pixel_format: "BGR8"  # Use BGR8 for color cameras like IDS U3-3680XCP-C
  
  resolution:
    width: 2592      # Full resolution
    height: 1944
    # Or use reduced resolution for better performance:
    # width: 1296
    # height: 972
  
  framerate: 30
  exposure: auto     # or value in microseconds
  gain: 1.0
  
  left_camera:
    device_id: 0     # USB camera index or serial number
  
  right_camera:
    device_id: 1

calibration:
  checkerboard:
    rows: 9          # Internal corners (not squares)
    cols: 6
    square_size: 25  # millimeters
  
  capture:
    num_images: 20   # Minimum for good calibration
    delay: 1.0       # Seconds between captures
```

### Stereo Configuration (`config/stereo_config.yaml`)

```yaml
stereo:
  baseline_mm: 60.0  # MUST BE MEASURED - distance between camera centers
  
  algorithm: "SGBM"  # or "BM" for faster processing
  
  sgbm:
    num_disparities: 128    # Must be divisible by 16
    block_size: 11          # Odd number, 5-21
    P1: 600
    P2: 2400
    uniqueness_ratio: 10
    speckle_window_size: 100
    speckle_range: 32
  
  post_process:
    use_wls_filter: true    # Better quality, slower
    wls_lambda: 8000
    wls_sigma: 1.5
    median_blur: 5
  
  visualization:
    colormap: "TURBO"       # JET, TURBO, HSV, VIRIDIS, etc.
    min_distance_m: 0.3
    max_distance_m: 10.0
    measurement_points: 9   # Grid of distance measurements
```

## Color Camera Support

### Pixel Format Configuration

The IDS U3-3680XCP-C cameras are **color cameras** with a Bayer color filter. The system supports multiple pixel formats:

| Format | Type | Channels | Description | Use Case |
|--------|------|----------|-------------|----------|
| **BGR8** | Color | 3 | 8-bit color, OpenCV native format | **Default for color cameras** |
| **RGB8** | Color | 3 | 8-bit color, standard RGB | Alternative color format |
| **BayerRG8** | Raw | 1 | Raw Bayer pattern (needs demosaicing) | Advanced processing |
| **Mono8** | Grayscale | 1 | 8-bit monochrome | For grayscale-only applications |

**Recommended setting**: Use **`BGR8`** for color cameras as it's the native format for OpenCV and provides the best performance.

Configure the pixel format in `config/camera_config.yaml`:

```yaml
cameras:
  pixel_format: "BGR8"  # Use BGR8 for IDS U3-3680XCP-C color cameras
```

### Color vs. Grayscale for Stereo Vision

**Note**: OpenCV's stereo matching algorithms (SGBM, BM) internally convert images to grayscale for disparity computation. However, capturing in color provides these benefits:

- **Better visualization**: Color images are easier to interpret
- **Post-processing**: Color information can be used for segmentation or filtering
- **Recording**: Captured images maintain full color information
- **Debugging**: Easier to identify objects and verify calibration

The stereo depth computation will work correctly with both color and grayscale inputs.

## Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save current frame and depth map |
| `c` | Cycle through colormaps |
| `f` | Toggle WLS filtering |
| `m` | Toggle measurement grid |
| `x` | Toggle crosshair |
| `+`/`-` | Adjust number of disparities |
| `[`/`]` | Adjust block size |
| `p` | Pause/resume |
| `h` | Show help |

## Measuring Baseline Distance

The **baseline** is the distance between the optical centers of the two cameras. This is critical for accurate depth measurement.

### How to Measure:
1. Measure the distance between the centers of the two lenses
2. Use a caliper or ruler for precision
3. Measure in millimeters
4. Update `baseline_mm` in `config/stereo_config.yaml`

**Typical Range**: 50-120mm for desktop stereo rigs

**Impact on Performance**:
- **Larger baseline**: Better accuracy at long distances, larger minimum distance
- **Smaller baseline**: Shorter minimum distance, reduced accuracy at far ranges

## Camera Calibration Theory

### Intrinsic Parameters
Describe the camera's internal characteristics:
- **Camera Matrix (K)**: Contains focal length (fx, fy) and principal point (cx, cy)
- **Distortion Coefficients**: Correct for lens distortion (radial and tangential)

### Extrinsic Parameters
Describe the relationship between cameras:
- **Rotation Matrix (R)**: 3x3 matrix describing rotation between cameras
- **Translation Vector (T)**: 3D vector describing translation (includes baseline)

### Estimated Intrinsic Parameters

For IDS U3-3680XCP-C at full resolution (2592x1944):

| Lens Focal Length | fx, fy (pixels) | Use Case |
|-------------------|-----------------|----------|
| 6mm | ~1620 | Wide field of view |
| 8mm | ~2160 | Standard |
| 12mm | ~3240 | Narrow, long range |

**Note**: These are estimates. Actual calibration is REQUIRED for accurate depth measurement.

## Depth Calculation

The system uses the standard stereo vision formula:

```
Z = (f × B) / d

Where:
- Z = distance to object (depth)
- f = focal length in pixels
- B = baseline (distance between cameras)
- d = disparity in pixels
```

### Distance Ranges

With typical setup (60mm baseline, 8mm lens, 1296x972 resolution):
- **Minimum distance**: ~30-50 cm
- **Maximum accurate distance**: ~5-10 m
- **Optimal range**: 0.5-3 m

## Troubleshooting

### Cameras Not Detected
- Check USB connections (use USB 3.0 ports)
- Verify device IDs: `ls /dev/video*` (Linux)
- Try different USB controllers
- Ensure no other application is using cameras
- Check camera permissions (Linux): `sudo usermod -a -G video $USER`

### Poor Calibration Results (High Error)
- Capture more images (30+ recommended)
- Improve image coverage (all areas of field of view)
- Use better lighting
- Ensure checkerboard is flat and sharp
- Check for motion blur
- Verify correct checkerboard dimensions in config

### Low Frame Rate
- Reduce resolution in `camera_config.yaml`
- Disable WLS filtering
- Reduce `num_disparities` in stereo config
- Use "BM" algorithm instead of "SGBM"
- Close other applications
- Use more powerful hardware

### Inaccurate Depth Measurements
- Verify calibration quality (reprojection error < 1.0)
- Measure and update baseline distance accurately
- Check stereo calibration error
- Ensure both cameras have similar exposure/brightness
- Verify cameras are rigidly mounted (no movement)
- Check that scene has sufficient texture

### No Depth in Certain Areas
- Increase `num_disparities` for far objects
- Ensure adequate lighting
- Add texture to scene (plain surfaces don't work well)
- Check disparity range matches scene depth
- Verify both cameras have same field of view

## File Structure

```
ids_stereo_vision/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore patterns
│
├── config/                           # Configuration files
│   ├── camera_config.yaml           # Camera settings
│   └── stereo_config.yaml           # Stereo processing settings
│
├── calibration/                      # Calibration scripts
│   ├── calibrate_single_camera.py   # Single camera calibration
│   ├── calibrate_stereo.py          # Stereo calibration
│   └── calibration_images/          # Captured calibration images
│       ├── left/                    # Left camera images
│       └── right/                   # Right camera images
│
├── calibration_data/                 # Saved calibration data
│   ├── left_camera_calibration.npz  # Left camera parameters
│   ├── right_camera_calibration.npz # Right camera parameters
│   └── stereo_calibration.npz       # Stereo parameters
│
├── src/                              # Core source code
│   ├── __init__.py
│   ├── camera_interface.py          # Camera communication
│   ├── stereo_processor.py          # Stereo processing pipeline
│   ├── depth_visualizer.py          # Visualization and overlays
│   └── utils.py                     # Utility functions
│
└── scripts/                          # Executable scripts
    ├── capture_calibration_images.py # Capture calibration images
    ├── test_cameras.py              # Test camera connections
    └── run_stereo_vision.py         # Main application
```

## Algorithm Details

### Stereo Matching Algorithms

#### StereoBM (Block Matching)
- **Speed**: Fast (~30-60 fps)
- **Quality**: Good for textured scenes
- **Best for**: Real-time applications, simple scenes

#### StereoSGBM (Semi-Global Block Matching)
- **Speed**: Moderate (~15-30 fps)
- **Quality**: Excellent, handles textureless regions better
- **Best for**: High-quality depth maps, complex scenes

### Post-Processing

#### WLS (Weighted Least Squares) Filter
- Refines disparity map using image edges
- Fills in holes and reduces noise
- Improves depth map quality significantly
- Cost: ~20-30% performance impact

#### Median Blur
- Removes salt-and-pepper noise
- Fast and effective
- Minimal performance impact

## Performance Optimization

### For Better FPS:
1. Reduce resolution to 1296x972 or 648x486
2. Use StereoBM instead of StereoSGBM
3. Disable WLS filtering
4. Reduce `num_disparities` to 64 or 96
5. Increase `block_size` to 15 or 21

### For Better Quality:
1. Use full resolution (2592x1944)
2. Use StereoSGBM algorithm
3. Enable WLS filtering
4. Use `num_disparities` of 128 or 160
5. Use smaller `block_size` (5-11)

## Future Improvements

- [ ] 3D point cloud generation and export (PLY format)
- [ ] Object detection and distance measurement
- [ ] Video recording with depth
- [ ] GPU acceleration (CUDA)
- [ ] Multiple stereo matching algorithms
- [ ] Automatic parameter tuning
- [ ] ROS integration
- [ ] Web interface for remote viewing
- [ ] Depth-based segmentation

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- OpenCV team for excellent computer vision library
- IDS Imaging for camera hardware and SDK
- Computer vision community for stereo vision algorithms

## References

- [OpenCV Stereo Vision Documentation](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- [IDS Software Suite](https://en.ids-imaging.com/downloads.html)
- [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)

## Support

For issues and questions:
- Check the Troubleshooting section
- Review configuration files
- Verify calibration quality
- Open an issue on GitHub

## Camera Specifications Reference

### IDS U3-3680XCP-C-HQ

| Specification | Value |
|---------------|-------|
| Sensor | ON Semiconductor AR0521 |
| Resolution | 2592 x 1944 (5.04 MP) |
| Sensor Size | 5.702 x 4.277 mm (1/2.5") |
| Pixel Size | 2.2 x 2.2 µm |
| Interface | USB 3.0 (USB3 Vision) |
| Frame Rate | Up to 49 fps @ full res |
| Lens Mount | C-mount |
| Shutter | Global shutter |

---

**Happy Stereo Vision Processing! 📷📷📐**