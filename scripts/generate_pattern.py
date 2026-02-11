#!/usr/bin/env python3
"""
Generate Chessboard Calibration Pattern
Creates a printable PDF or image of a chessboard pattern for camera calibration.
"""

import sys
import argparse
import numpy as np
import cv2


def generate_chessboard(rows, cols, square_size_mm, dpi=300, output_file="calibration_pattern.pdf"):
    """
    Generate a chessboard calibration pattern.
    
    Args:
        rows: Number of squares vertically
        cols: Number of squares horizontally
        square_size_mm: Size of each square in millimeters
        dpi: Dots per inch for output
        output_file: Output filename (.png or .pdf)
    """
    # Calculate pixels per mm
    pixels_per_mm = dpi / 25.4
    square_size_pixels = int(square_size_mm * pixels_per_mm)
    
    # Image dimensions
    img_width = cols * square_size_pixels
    img_height = rows * square_size_pixels
    
    print(f"Generating chessboard pattern:")
    print(f"  Squares: {cols} × {rows}")
    print(f"  Square size: {square_size_mm}mm ({square_size_pixels} pixels)")
    print(f"  Total size: {cols * square_size_mm}mm × {rows * square_size_mm}mm")
    print(f"  Image size: {img_width} × {img_height} pixels")
    print(f"  DPI: {dpi}")
    
    # Create image
    img = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Fill squares
    for row in range(rows):
        for col in range(cols):
            # Checkerboard pattern: alternate starting with black in top-left
            if (row + col) % 2 == 0:
                y_start = row * square_size_pixels
                y_end = (row + 1) * square_size_pixels
                x_start = col * square_size_pixels
                x_end = (col + 1) * square_size_pixels
                img[y_start:y_end, x_start:x_end] = 255  # White square
    
    # Add border
    border_pixels = int(square_size_pixels * 0.5)
    img_with_border = np.zeros(
        (img_height + 2 * border_pixels, img_width + 2 * border_pixels),
        dtype=np.uint8
    )
    img_with_border[border_pixels:-border_pixels, border_pixels:-border_pixels] = img
    img_with_border[:border_pixels, :] = 255  # White border
    img_with_border[-border_pixels:, :] = 255
    img_with_border[:, :border_pixels] = 255
    img_with_border[:, -border_pixels:] = 255
    
    # Add text information
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = square_size_pixels / 100.0
    thickness = max(1, int(square_size_pixels / 50))
    
    text_lines = [
        f"Calibration Pattern: {cols}×{rows} squares",
        f"Square size: {square_size_mm}mm",
        f"Internal corners: {cols-1}×{rows-1}",
        f"Print at 100% scale (no fit-to-page)"
    ]
    
    y_offset = border_pixels // 4
    for i, line in enumerate(text_lines):
        text_size = cv2.getTextSize(line, font, font_scale * 0.6, thickness)[0]
        x_pos = (img_with_border.shape[1] - text_size[0]) // 2
        y_pos = y_offset + int(i * text_size[1] * 1.5)
        
        # Add white background for text
        cv2.rectangle(
            img_with_border,
            (x_pos - 10, y_pos - text_size[1] - 5),
            (x_pos + text_size[0] + 10, y_pos + 5),
            255,
            -1
        )
        
        cv2.putText(
            img_with_border,
            line,
            (x_pos, y_pos),
            font,
            font_scale * 0.6,
            0,
            thickness
        )
    
    # Save image
    if output_file.lower().endswith('.pdf'):
        # For PDF, we'll save as high-quality PNG first, then note to convert
        png_file = output_file.replace('.pdf', '.png')
        cv2.imwrite(png_file, img_with_border)
        print(f"\n✓ Generated: {png_file}")
        print(f"\nTo convert to PDF:")
        print(f"  - Use 'img2pdf {png_file} -o {output_file}'")
        print(f"  - Or open in image viewer and 'Print to PDF'")
        print(f"  - Ensure 100% scale, no 'fit to page'")
    else:
        cv2.imwrite(output_file, img_with_border)
        print(f"\n✓ Generated: {output_file}")
    
    # Create a reference measurement image
    ref_img = img_with_border.copy()
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
    
    # Draw measurement lines
    center_x = ref_img.shape[1] // 2
    center_y = ref_img.shape[0] // 2
    
    # Horizontal line
    cv2.line(ref_img, 
             (border_pixels, center_y), 
             (border_pixels + square_size_pixels, center_y),
             (0, 0, 255), thickness * 2)
    cv2.line(ref_img, 
             (border_pixels, center_y - 10), 
             (border_pixels, center_y + 10),
             (0, 0, 255), thickness * 2)
    cv2.line(ref_img, 
             (border_pixels + square_size_pixels, center_y - 10), 
             (border_pixels + square_size_pixels, center_y + 10),
             (0, 0, 255), thickness * 2)
    
    # Add measurement text
    cv2.putText(ref_img, f"{square_size_mm}mm",
                (border_pixels + square_size_pixels // 4, center_y - 20),
                font, font_scale * 0.8, (0, 0, 255), thickness * 2)
    
    ref_file = output_file.replace('.png', '_reference.png').replace('.pdf', '_reference.png')
    cv2.imwrite(ref_file, ref_img)
    print(f"✓ Reference with measurements: {ref_file}")
    
    print(f"\nPRINTING INSTRUCTIONS:")
    print(f"1. Open {output_file} in your PDF viewer or image viewer")
    print(f"2. Print settings:")
    print(f"   - Scale: 100% (ACTUAL SIZE)")
    print(f"   - Do NOT select 'Fit to page'")
    print(f"   - Quality: Highest/Best")
    print(f"   - Paper: A4 (or larger)")
    print(f"3. After printing, verify with ruler:")
    print(f"   - Measure one square: should be exactly {square_size_mm}mm")
    print(f"   - If not exact, adjust printer settings and reprint")
    print(f"4. Mount on flat, rigid surface (cardboard backing)")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate chessboard calibration pattern for printing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 8×6 pattern with 10mm squares
  python generate_pattern.py --rows 6 --cols 8 --size 10

  # Generate 7×5 pattern with 15mm squares
  python generate_pattern.py --rows 5 --cols 7 --size 15 --output my_pattern.png

  # High DPI for laser printer
  python generate_pattern.py --rows 7 --cols 9 --size 12 --dpi 600

Note: 
  - Rows and cols refer to number of SQUARES (not corners)
  - For N×M squares, you get (N-1)×(M-1) internal corners
  - Example: 8×6 squares = 7×5 corners for calibration
        """
    )
    
    parser.add_argument(
        '--rows',
        type=int,
        required=True,
        help='Number of squares vertically (height)'
    )
    
    parser.add_argument(
        '--cols',
        type=int,
        required=True,
        help='Number of squares horizontally (width)'
    )
    
    parser.add_argument(
        '--size',
        type=float,
        required=True,
        help='Size of each square in millimeters'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Printer DPI (default: 300)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='calibration_pattern.png',
        help='Output filename (.png or .pdf)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.rows < 3 or args.cols < 3:
        print("Error: Minimum 3×3 squares required")
        return 1
    
    if args.size <= 0:
        print("Error: Square size must be positive")
        return 1
    
    if args.dpi < 72:
        print("Error: DPI must be at least 72")
        return 1
    
    # Calculate A4 size limits
    a4_width_mm = 210
    a4_height_mm = 297
    total_width = args.cols * args.size
    total_height = args.rows * args.size
    
    if total_width > a4_width_mm or total_height > a4_height_mm:
        print(f"Warning: Pattern size ({total_width:.0f}×{total_height:.0f}mm) may be too large for A4")
        print(f"         A4 paper is {a4_width_mm}×{a4_height_mm}mm")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            return 1
    
    # Generate pattern
    generate_chessboard(args.rows, args.cols, args.size, args.dpi, args.output)
    
    print(f"\n{'='*60}")
    print(f"NEXT STEPS:")
    print(f"{'='*60}")
    print(f"1. Print the pattern at 100% scale")
    print(f"2. Verify square size with ruler ({args.size}mm)")
    print(f"3. Update config/stereo_config.yaml:")
    print(f"   width: {args.cols - 1}  # internal corners")
    print(f"   height: {args.rows - 1}")
    print(f"   square_size: {args.size}")
    print(f"4. Run: python scripts/capture_calibration_images.py")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
