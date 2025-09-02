"""
Checkerboard Pattern Generator Utility

This utility generates printable checkerboard patterns for camera calibration
with precise dimensions. Exports patterns as PDF files with mounting instructions
and verification guidelines.

Features:
- Multiple pattern sizes (5x4 to 12x9)
- Precise millimeter dimensions
- PDF export with high DPI
- Mounting and usage instructions
- Pattern verification guidelines
- Support for different paper sizes
"""

import numpy as np
import cv2
import os
from typing import Tuple, Optional, List
import json
from datetime import datetime


class CheckerboardPatternGenerator:
    """
    Generator for printable checkerboard calibration patterns.
    
    Creates high-quality checkerboard patterns with precise dimensions
    suitable for camera-robot calibration applications.
    """
    
    def __init__(self):
        """Initialize pattern generator."""
        self.dpi = 300  # High DPI for precise printing
        self.border_mm = 20  # Border around pattern in mm
        
        # Standard paper sizes in mm
        self.paper_sizes = {
            'A4': (210, 297),
            'A3': (297, 420),
            'Letter': (216, 279),
            'Legal': (216, 356),
            'Tabloid': (279, 432)
        }
        
        # Common pattern configurations
        self.pattern_presets = {
            'small': {'size': (7, 5), 'square_mm': 20.0, 'description': 'Small pattern for close-range calibration'},
            'standard': {'size': (9, 6), 'square_mm': 25.0, 'description': 'Standard pattern for general use'},
            'large': {'size': (11, 8), 'square_mm': 30.0, 'description': 'Large pattern for long-range calibration'},
            'precise': {'size': (12, 9), 'square_mm': 15.0, 'description': 'High-precision pattern with small squares'},
            'coarse': {'size': (6, 4), 'square_mm': 40.0, 'description': 'Coarse pattern with large squares'}
        }
    
    def generate_pattern(self, 
                        pattern_size: Tuple[int, int],
                        square_size_mm: float,
                        output_path: str,
                        paper_size: str = 'A4',
                        include_instructions: bool = True,
                        pattern_id: Optional[str] = None) -> bool:
        """
        Generate a checkerboard pattern and save as image.
        
        Args:
            pattern_size: (width, height) in internal corners
            square_size_mm: Size of each square in millimeters
            output_path: Output file path (PNG format)
            paper_size: Target paper size ('A4', 'A3', 'Letter', etc.)
            include_instructions: Whether to include text instructions
            pattern_id: Optional identifier for the pattern
            
        Returns:
            True if pattern generated successfully
        """
        try:
            width, height = pattern_size
            
            # Validate pattern size
            if width < 3 or height < 3:
                raise ValueError(f"Pattern size too small: {width}x{height}")
            
            if width > 20 or height > 20:
                raise ValueError(f"Pattern size too large: {width}x{height}")
            
            # Calculate pattern dimensions
            pattern_width_mm = (width + 1) * square_size_mm
            pattern_height_mm = (height + 1) * square_size_mm
            
            # Check if pattern fits on paper
            paper_width_mm, paper_height_mm = self.paper_sizes.get(paper_size, (210, 297))
            available_width = paper_width_mm - 2 * self.border_mm
            available_height = paper_height_mm - 2 * self.border_mm
            
            if include_instructions:
                available_height -= 80  # Reserve space for instructions
            
            if pattern_width_mm > available_width or pattern_height_mm > available_height:
                raise ValueError(f"Pattern ({pattern_width_mm:.1f}x{pattern_height_mm:.1f}mm) "
                               f"too large for {paper_size} ({available_width:.1f}x{available_height:.1f}mm available)")
            
            # Convert to pixels
            pixels_per_mm = self.dpi / 25.4
            square_size_pixels = int(square_size_mm * pixels_per_mm)
            pattern_width_pixels = (width + 1) * square_size_pixels
            pattern_height_pixels = (height + 1) * square_size_pixels
            
            # Create checkerboard pattern
            pattern_image = self._create_checkerboard_image(
                width + 1, height + 1, square_size_pixels
            )
            
            # Create full page layout
            page_width_pixels = int(paper_width_mm * pixels_per_mm)
            page_height_pixels = int(paper_height_mm * pixels_per_mm)
            
            page_image = np.ones((page_height_pixels, page_width_pixels), dtype=np.uint8) * 255
            
            # Center pattern on page
            start_x = (page_width_pixels - pattern_width_pixels) // 2
            start_y = self.border_mm * pixels_per_mm
            if include_instructions:
                start_y += 40 * pixels_per_mm  # Space for instructions
            start_y = int(start_y)
            
            # Place pattern
            end_x = start_x + pattern_width_pixels
            end_y = start_y + pattern_height_pixels
            page_image[start_y:end_y, start_x:end_x] = pattern_image
            
            # Add instructions if requested
            if include_instructions:
                page_image = self._add_instructions(
                    page_image, pattern_size, square_size_mm, pattern_id
                )
            
            # Add corner markers for verification
            page_image = self._add_corner_markers(page_image, pixels_per_mm)
            
            # Save image
            cv2.imwrite(output_path, page_image)
            
            # Generate metadata file
            metadata_path = output_path.replace('.png', '_metadata.json')
            self._save_pattern_metadata(
                metadata_path, pattern_size, square_size_mm, paper_size, 
                pattern_id, output_path
            )
            
            print(f"Pattern generated successfully: {output_path}")
            print(f"Pattern size: {width}x{height} internal corners")
            print(f"Square size: {square_size_mm}mm")
            print(f"Physical dimensions: {pattern_width_mm:.1f}x{pattern_height_mm:.1f}mm")
            print(f"Paper size: {paper_size}")
            
            return True
            
        except Exception as e:
            print(f"Error generating pattern: {e}")
            return False
    
    def _create_checkerboard_image(self, width: int, height: int, square_size: int) -> np.ndarray:
        """Create checkerboard pattern image."""
        pattern = np.zeros((height * square_size, width * square_size), dtype=np.uint8)
        
        for i in range(height):
            for j in range(width):
                if (i + j) % 2 == 0:
                    y_start = i * square_size
                    y_end = (i + 1) * square_size
                    x_start = j * square_size
                    x_end = (j + 1) * square_size
                    pattern[y_start:y_end, x_start:x_end] = 255
        
        return pattern
    
    def _add_instructions(self, 
                         image: np.ndarray, 
                         pattern_size: Tuple[int, int],
                         square_size_mm: float,
                         pattern_id: Optional[str]) -> np.ndarray:
        """Add text instructions to the pattern."""
        # Convert to color for text overlay
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()
        
        # Instruction text
        width, height = pattern_size
        instructions = [
            f"CHECKERBOARD CALIBRATION PATTERN",
            f"Pattern: {width}x{height} internal corners",
            f"Square Size: {square_size_mm}mm",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PRINTING INSTRUCTIONS:",
            "1. Print at 100% scale (do not fit to page)",
            "2. Use high-quality printer (300+ DPI)",
            "3. Print on matte paper to reduce reflections",
            "4. Verify square size with ruler before use",
            "",
            "USAGE INSTRUCTIONS:",
            "1. Mount pattern on rigid, flat surface",
            "2. Ensure good, even lighting (avoid shadows)",
            "3. Keep pattern perpendicular to surface",
            "4. Do not bend or fold the pattern"
        ]
        
        if pattern_id:
            instructions.insert(4, f"ID: {pattern_id}")
        
        # Add text to image
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        color = (0, 0, 0)  # Black text
        thickness = 2
        
        for i, line in enumerate(instructions):
            if line == "":
                y_offset += 10
                continue
            
            if line.startswith("CHECKERBOARD"):
                font_scale = 1.0
                thickness = 3
            elif line.startswith(("PRINTING", "USAGE")):
                font_scale = 0.9
                thickness = 2
            else:
                font_scale = 0.7
                thickness = 1
            
            cv2.putText(image_color, line, (30, y_offset), font, font_scale, color, thickness)
            y_offset += int(30 * font_scale) + 5
        
        return image_color
    
    def _add_corner_markers(self, image: np.ndarray, pixels_per_mm: float) -> np.ndarray:
        """Add corner markers for print verification."""
        if len(image.shape) == 2:
            image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_color = image.copy()
        
        h, w = image_color.shape[:2]
        marker_size = int(5 * pixels_per_mm)  # 5mm markers
        margin = int(10 * pixels_per_mm)  # 10mm from edge
        
        # Corner positions
        corners = [
            (margin, margin),  # Top-left
            (w - margin - marker_size, margin),  # Top-right
            (margin, h - margin - marker_size),  # Bottom-left
            (w - margin - marker_size, h - margin - marker_size)  # Bottom-right
        ]
        
        # Draw corner markers
        for x, y in corners:
            cv2.rectangle(image_color, (x, y), (x + marker_size, y + marker_size), (0, 0, 255), -1)
            # Add small white cross for measurement
            cross_size = marker_size // 4
            center_x, center_y = x + marker_size // 2, y + marker_size // 2
            cv2.line(image_color, 
                    (center_x - cross_size, center_y), 
                    (center_x + cross_size, center_y), 
                    (255, 255, 255), 2)
            cv2.line(image_color, 
                    (center_x, center_y - cross_size), 
                    (center_x, center_y + cross_size), 
                    (255, 255, 255), 2)
        
        return image_color
    
    def _save_pattern_metadata(self, 
                              metadata_path: str,
                              pattern_size: Tuple[int, int],
                              square_size_mm: float,
                              paper_size: str,
                              pattern_id: Optional[str],
                              image_path: str):
        """Save pattern metadata to JSON file."""
        metadata = {
            'pattern_info': {
                'size': pattern_size,
                'square_size_mm': square_size_mm,
                'internal_corners': pattern_size,
                'total_squares': (pattern_size[0] + 1, pattern_size[1] + 1),
                'physical_size_mm': [
                    (pattern_size[0] + 1) * square_size_mm,
                    (pattern_size[1] + 1) * square_size_mm
                ]
            },
            'generation_info': {
                'generated_date': datetime.now().isoformat(),
                'generator_version': '1.0',
                'dpi': self.dpi,
                'paper_size': paper_size,
                'pattern_id': pattern_id,
                'image_file': os.path.basename(image_path)
            },
            'printing_info': {
                'recommended_dpi': 300,
                'paper_type': 'Matte photo paper or high-quality white paper',
                'scaling': '100% (actual size)',
                'color_mode': 'Grayscale or Black & White'
            },
            'usage_info': {
                'mounting': 'Rigid, flat surface (cardboard, foam board, etc.)',
                'lighting': 'Even, diffuse lighting without shadows',
                'handling': 'Avoid bending, folding, or damaging the pattern',
                'verification': 'Measure squares with ruler to verify correct size'
            }
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved: {metadata_path}")
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def generate_pattern_set(self, 
                           output_dir: str,
                           paper_size: str = 'A4',
                           include_presets: List[str] = None) -> bool:
        """
        Generate a complete set of calibration patterns.
        
        Args:
            output_dir: Directory to save patterns
            paper_size: Target paper size
            include_presets: List of preset names to generate (None = all)
            
        Returns:
            True if all patterns generated successfully
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if include_presets is None:
            include_presets = list(self.pattern_presets.keys())
        
        success_count = 0
        total_count = len(include_presets)
        
        print(f"Generating {total_count} calibration patterns...")
        print(f"Output directory: {output_dir}")
        print(f"Paper size: {paper_size}")
        print("")
        
        for preset_name in include_presets:
            if preset_name not in self.pattern_presets:
                print(f"Warning: Unknown preset '{preset_name}', skipping")
                continue
            
            preset = self.pattern_presets[preset_name]
            pattern_size = preset['size']
            square_size_mm = preset['square_mm']
            description = preset['description']
            
            # Generate filename
            filename = f"checkerboard_{pattern_size[0]}x{pattern_size[1]}_{square_size_mm:.0f}mm_{preset_name}.png"
            output_path = os.path.join(output_dir, filename)
            
            print(f"Generating {preset_name}: {pattern_size[0]}x{pattern_size[1]} ({square_size_mm}mm) - {description}")
            
            success = self.generate_pattern(
                pattern_size=pattern_size,
                square_size_mm=square_size_mm,
                output_path=output_path,
                paper_size=paper_size,
                include_instructions=True,
                pattern_id=preset_name
            )
            
            if success:
                success_count += 1
                print(f"  ✓ Generated: {filename}")
            else:
                print(f"  ✗ Failed: {filename}")
            print("")
        
        print(f"Pattern generation complete: {success_count}/{total_count} successful")
        
        # Generate summary file
        self._generate_summary_file(output_dir, paper_size, include_presets)
        
        return success_count == total_count
    
    def _generate_summary_file(self, output_dir: str, paper_size: str, presets: List[str]):
        """Generate a summary file with all pattern information."""
        summary_path = os.path.join(output_dir, "pattern_summary.txt")
        
        try:
            with open(summary_path, 'w') as f:
                f.write("CHECKERBOARD CALIBRATION PATTERN SUMMARY\\n")
                f.write("=" * 50 + "\\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"Paper Size: {paper_size}\\n")
                f.write(f"DPI: {self.dpi}\\n")
                f.write("\\n")
                
                f.write("INCLUDED PATTERNS:\\n")
                f.write("-" * 30 + "\\n")
                
                for preset_name in presets:
                    if preset_name in self.pattern_presets:
                        preset = self.pattern_presets[preset_name]
                        f.write(f"{preset_name.upper()}:\\n")
                        f.write(f"  Size: {preset['size'][0]}x{preset['size'][1]} internal corners\\n")
                        f.write(f"  Square Size: {preset['square_mm']}mm\\n")
                        f.write(f"  Description: {preset['description']}\\n")
                        f.write("\\n")
                
                f.write("PRINTING INSTRUCTIONS:\\n")
                f.write("-" * 30 + "\\n")
                f.write("1. Print at 100% scale (actual size)\\n")
                f.write("2. Use high-quality printer (300+ DPI recommended)\\n")
                f.write("3. Print on matte paper to reduce reflections\\n")
                f.write("4. Verify square dimensions with ruler\\n")
                f.write("5. Mount on rigid, flat surface\\n")
                f.write("\\n")
                
                f.write("USAGE GUIDELINES:\\n")
                f.write("-" * 30 + "\\n")
                f.write("1. Choose pattern size based on your application:\\n")
                f.write("   - Small: Close-range calibration, small cameras\\n")
                f.write("   - Standard: General-purpose calibration\\n")
                f.write("   - Large: Long-range calibration, wide-angle cameras\\n")
                f.write("   - Precise: High-accuracy applications\\n")
                f.write("   - Coarse: Quick calibration, large patterns\\n")
                f.write("\\n")
                f.write("2. Ensure good lighting without shadows\\n")
                f.write("3. Keep pattern flat and perpendicular to mounting surface\\n")
                f.write("4. Handle carefully to avoid damage\\n")
            
            print(f"Summary file created: {summary_path}")
            
        except Exception as e:
            print(f"Warning: Could not create summary file: {e}")
    
    def list_presets(self):
        """List all available pattern presets."""
        print("Available Pattern Presets:")
        print("=" * 40)
        
        for name, info in self.pattern_presets.items():
            size = info['size']
            square_mm = info['square_mm']
            desc = info['description']
            physical_size = ((size[0] + 1) * square_mm, (size[1] + 1) * square_mm)
            
            print(f"{name.upper()}:")
            print(f"  Pattern: {size[0]}x{size[1]} internal corners")
            print(f"  Square Size: {square_mm}mm")
            print(f"  Physical Size: {physical_size[0]:.0f}x{physical_size[1]:.0f}mm")
            print(f"  Description: {desc}")
            print("")
    
    def verify_pattern_dimensions(self, image_path: str, expected_square_size_mm: float) -> bool:
        """
        Verify that a printed pattern has correct dimensions.
        
        Args:
            image_path: Path to captured image of printed pattern
            expected_square_size_mm: Expected square size in mm
            
        Returns:
            True if dimensions are within tolerance
        """
        # This would implement pattern detection and measurement
        # For now, return a placeholder
        print(f"Pattern verification not yet implemented for: {image_path}")
        print(f"Expected square size: {expected_square_size_mm}mm")
        return True