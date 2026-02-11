#!/usr/bin/env python3
"""
Calibration Pattern Calculator
Helps determine the optimal chessboard pattern size for your specific setup.
"""

import sys
import argparse
import math


def calculate_fov(sensor_width_mm, sensor_height_mm, focal_length_mm, distance_mm):
    """Calculate field of view at a given distance."""
    fov_width = (sensor_width_mm * distance_mm) / focal_length_mm
    fov_height = (sensor_height_mm * distance_mm) / focal_length_mm
    return fov_width, fov_height


def calculate_pixel_size_at_distance(pixel_size_um, focal_length_mm, distance_mm):
    """Calculate the physical size represented by one pixel at the pattern."""
    return (pixel_size_um / 1000.0) * distance_mm / focal_length_mm


def recommend_patterns(focal_length, distance, paper_width=210, paper_height=297):
    """
    Recommend calibration patterns for IDS u3-3680xcp-c cameras.
    
    Args:
        focal_length: Lens focal length in mm
        distance: Distance to calibration pattern in mm
        paper_width: Paper width in mm (default: A4 width)
        paper_height: Paper height in mm (default: A4 height)
    """
    # IDS u3-3680xcp-c specifications
    SENSOR_WIDTH = 5.702  # mm
    SENSOR_HEIGHT = 4.277  # mm
    PIXEL_SIZE = 2.2  # micrometers
    
    print("\n" + "=" * 80)
    print(f"CALIBRATION PATTERN RECOMMENDATIONS")
    print("=" * 80)
    print(f"\nCamera: IDS u3-3680xcp-c")
    print(f"  Sensor: {SENSOR_WIDTH}mm × {SENSOR_HEIGHT}mm")
    print(f"  Resolution: 2592 × 1944 pixels")
    print(f"  Pixel size: {PIXEL_SIZE}µm")
    print(f"\nSetup:")
    print(f"  Lens focal length: {focal_length}mm")
    print(f"  Distance to pattern: {distance}mm ({distance/10:.1f}cm)")
    print(f"  Paper size: {paper_width}mm × {paper_height}mm")
    
    # Calculate FOV
    fov_width, fov_height = calculate_fov(SENSOR_WIDTH, SENSOR_HEIGHT, focal_length, distance)
    
    print(f"\nField of View at {distance}mm:")
    print(f"  Width: {fov_width:.1f}mm")
    print(f"  Height: {fov_height:.1f}mm")
    
    # Check paper fit
    if fov_width < paper_width or fov_height < paper_height:
        print(f"\n  ⚠️  WARNING: Paper is larger than FOV!")
        print(f"      You may need to move farther or use a wider-angle lens.")
        # Use smaller dimension
        usable_width = min(fov_width * 0.9, paper_width)
        usable_height = min(fov_height * 0.9, paper_height)
    else:
        print(f"\n  ✓ Paper fits in FOV")
        usable_width = paper_width
        usable_height = paper_height
    
    # Calculate pixel size at distance
    pixel_size_at_dist = calculate_pixel_size_at_distance(PIXEL_SIZE, focal_length, distance)
    print(f"\nPixel size at pattern: {pixel_size_at_dist:.3f}mm/pixel")
    
    # Optimal square size: 10-40 pixels per square (practical range)
    min_square = pixel_size_at_dist * 10
    optimal_min_square = pixel_size_at_dist * 15
    optimal_max_square = pixel_size_at_dist * 30
    max_square = pixel_size_at_dist * 50
    
    print(f"\nSquare size guidelines:")
    print(f"  Minimum viable: {min_square:.1f}mm (10 pixels/square)")
    print(f"  Optimal range: {optimal_min_square:.1f}mm - {optimal_max_square:.1f}mm (15-30 pixels/square)")
    print(f"  Maximum recommended: {max_square:.1f}mm (50 pixels/square)")
    
    # Generate pattern recommendations
    print(f"\n" + "=" * 80)
    print(f"RECOMMENDED PATTERNS (sorted by quality)")
    print("=" * 80)
    
    # Try different square sizes (practical range for printing)
    recommendations = []
    
    # Pattern should cover 65-75% of usable area
    target_coverage = 0.70
    
    # Use practical square sizes that can be printed accurately
    square_sizes_to_try = list(range(5, 26)) + list(range(26, 51, 2))  # 5-25mm every 1mm, then 26-50mm every 2mm
    
    for square_size in square_sizes_to_try:
        # Don't filter by min/max square size - let user decide based on quality score
        # Just warn if outside optimal range
        if square_size < min_square * 0.5:  # Too ridiculously small
            continue
        if square_size > max_square * 2:  # Too ridiculously large
            continue
        
        # Calculate pattern dimensions
        target_width = usable_width * target_coverage
        target_height = usable_height * target_coverage
        
        squares_h = int(target_height / square_size)
        squares_w = int(target_width / square_size)
        
        # Internal corners
        corners_h = squares_h - 1
        corners_w = squares_w - 1
        
        # Need at least 5×4 corners
        if corners_w < 5 or corners_h < 4:
            continue
        
        # Prefer more square-like patterns
        aspect_ratio = max(corners_w, corners_h) / min(corners_w, corners_h)
        if aspect_ratio > 2.0:
            continue
        
        total_width = squares_w * square_size
        total_height = squares_h * square_size
        
        # Check if fits on paper
        if total_width > paper_width or total_height > paper_height:
            continue
        
        pixels_per_square = square_size / pixel_size_at_dist
        num_corners = corners_w * corners_h
        
        # Quality score - prefer practical patterns
        target_pixels = 20.0  # Optimal pixels per square
        pixel_score = 1.0 - abs(pixels_per_square - target_pixels) / target_pixels
        pixel_score = max(0, min(1, pixel_score))
        
        # Prefer 30-70 corners (not too few, not too many)
        target_corners = 50
        corner_score = 1.0 - abs(num_corners - target_corners) / target_corners
        corner_score = max(0, min(1, corner_score))
        
        # Prefer square-ish patterns
        aspect_score = 1.0 - (aspect_ratio - 1.0) / 2.0
        aspect_score = max(0, aspect_score)
        
        # Penalize very small square sizes (hard to print)
        size_score = 1.0
        if square_size < 8:
            size_score = square_size / 8.0
        
        quality = (pixel_score * 0.3 + corner_score * 0.3 + aspect_score * 0.2 + size_score * 0.2)
        
        recommendations.append({
            'square_size': square_size,
            'corners_w': corners_w,
            'corners_h': corners_h,
            'squares_w': squares_w,
            'squares_h': squares_h,
            'total_width': total_width,
            'total_height': total_height,
            'pixels_per_square': pixels_per_square,
            'num_corners': num_corners,
            'quality': quality
        })
    
    # Sort by quality
    recommendations.sort(key=lambda x: x['quality'], reverse=True)
    
    # Show top recommendations
    if not recommendations:
        print("\n⚠️  No suitable patterns found for this configuration!")
        print("   Try adjusting distance or using a different focal length lens.")
        return None
    
    print(f"\nTop {min(5, len(recommendations))} Patterns:\n")
    
    for i, rec in enumerate(recommendations[:5], 1):
        # Check if square size is practical
        is_small = rec['square_size'] < 8
        is_large = rec['square_size'] > 20
        is_many_corners = rec['num_corners'] > 80
        
        print(f"{i}. Square size: {rec['square_size']}mm", end="")
        if is_small:
            print(" ⚠️ Small - may be hard to print accurately", end="")
        if is_large:
            print(" ⚠️ Large - fewer corners", end="")
        print()
        
        print(f"   Pattern: {rec['corners_w']}×{rec['corners_h']} corners "
              f"({rec['squares_w']}×{rec['squares_h']} squares)")
        print(f"   Print size: {rec['total_width']:.0f}mm × {rec['total_height']:.0f}mm")
        print(f"   Pixels per square: {rec['pixels_per_square']:.1f}")
        print(f"   Total corners: {rec['num_corners']}", end="")
        if is_many_corners:
            print(" (many - longer calibration)", end="")
        print()
        print(f"   Quality score: {rec['quality']:.2f}")
        
        # Mark the best one
        if i == 1:
            print(f"   ⭐ RECOMMENDED - Best balance of detection and accuracy")
        print()
    
    # Return best recommendation
    best = recommendations[0]
    
    print("=" * 80)
    print("SUMMARY FOR CONFIG FILE:")
    print("=" * 80)
    print(f"""
In your config/stereo_config.yaml, use:

system:
  calibration_pattern:
    type: "chessboard"
    width: {best['corners_w']}     # internal corners (horizontal)
    height: {best['corners_h']}    # internal corners (vertical)
    square_size: {best['square_size']}.0  # mm
""")
    
    print("=" * 80)
    print("PRINTING INSTRUCTIONS:")
    print("=" * 80)
    print(f"""
1. Pattern to print: {best['corners_w']}×{best['corners_h']} corners ({best['squares_w']}×{best['squares_h']} squares)
2. Square size: {best['square_size']}mm
3. Total pattern size: {best['total_width']:.0f}mm × {best['total_height']:.0f}mm

To create the pattern:

Option A - Online Generator:
  1. Visit: https://calib.io/pages/camera-calibration-pattern-generator
  2. Select "Checkerboard"
  3. Enter: {best['squares_w']}×{best['squares_h']} squares
  4. Square size: {best['square_size']}mm
  5. Download PDF and print on A4

Option B - OpenCV Python:
  python scripts/generate_pattern.py --rows {best['squares_h']} --cols {best['squares_w']} --size {best['square_size']}

Printing tips:
  • Use highest quality setting (no scaling!)
  • Print actual size (100%, no "fit to page")
  • Use thick paper or mount on cardboard
  • Ensure black squares are fully black
  • White squares should be clean white
  • Verify printed square size with ruler
""")
    
    print("=" * 80)
    
    return best


def main():
    parser = argparse.ArgumentParser(
        description="Calculate optimal calibration pattern for IDS u3-3680xcp-c cameras",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # For 6mm lens at 20cm
  python calculate_calibration_pattern.py --focal 6 --distance 200

  # For 8mm lens at 30cm on A4 paper
  python calculate_calibration_pattern.py --focal 8 --distance 300

  # For 12mm lens at 50cm
  python calculate_calibration_pattern.py --focal 12 --distance 500
        """
    )
    
    parser.add_argument(
        '--focal',
        type=float,
        required=True,
        help='Lens focal length in mm (check your C-mount lens)'
    )
    
    parser.add_argument(
        '--distance',
        type=float,
        required=True,
        help='Distance from camera to pattern in mm (e.g., 200 for 20cm)'
    )
    
    parser.add_argument(
        '--paper-width',
        type=float,
        default=210,
        help='Paper width in mm (default: 210 for A4)'
    )
    
    parser.add_argument(
        '--paper-height',
        type=float,
        default=297,
        help='Paper height in mm (default: 297 for A4)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.focal <= 0:
        print("Error: Focal length must be positive")
        return 1
    
    if args.distance <= 0:
        print("Error: Distance must be positive")
        return 1
    
    if args.distance < args.focal * 10:
        print(f"Warning: Distance ({args.distance}mm) is very close for {args.focal}mm lens")
        print(f"         Minimum recommended: {args.focal * 10}mm")
        print()
    
    # Calculate recommendations
    recommend_patterns(
        args.focal,
        args.distance,
        args.paper_width,
        args.paper_height
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
