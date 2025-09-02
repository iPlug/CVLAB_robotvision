#!/usr/bin/env python3
"""
Checkerboard Pattern Generator Script

This script generates printable checkerboard patterns for camera calibration.
Creates high-quality PDF-ready patterns with precise dimensions and instructions.

Usage:
    python generate_checkerboard_patterns.py [options]

Examples:
    # Generate standard 9x6 pattern
    python generate_checkerboard_patterns.py

    # Generate custom pattern
    python generate_checkerboard_patterns.py --pattern 8 6 --square-size 30.0

    # Generate complete pattern set
    python generate_checkerboard_patterns.py --generate-set

    # List available presets
    python generate_checkerboard_patterns.py --list-presets
"""

import sys
import os
import argparse

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.utils.pattern_generator import CheckerboardPatternGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate checkerboard calibration patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate standard pattern
    python generate_checkerboard_patterns.py
    
    # Custom pattern size and square size
    python generate_checkerboard_patterns.py --pattern 8 6 --square-size 30.0
    
    # Generate complete set for A3 paper
    python generate_checkerboard_patterns.py --generate-set --paper A3
    
    # Generate specific presets
    python generate_checkerboard_patterns.py --generate-set --presets standard large
        """
    )
    
    parser.add_argument(
        '--pattern',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        default=[9, 6],
        help='Pattern size in internal corners (default: 9 6)'
    )
    
    parser.add_argument(
        '--square-size',
        type=float,
        default=25.0,
        help='Size of each square in millimeters (default: 25.0)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: auto-generated)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='checkerboard_patterns',
        help='Output directory for pattern set (default: checkerboard_patterns)'
    )
    
    parser.add_argument(
        '--paper',
        type=str,
        choices=['A4', 'A3', 'Letter', 'Legal', 'Tabloid'],
        default='A4',
        help='Paper size (default: A4)'
    )
    
    parser.add_argument(
        '--no-instructions',
        action='store_true',
        help='Generate pattern without text instructions'
    )
    
    parser.add_argument(
        '--pattern-id',
        type=str,
        default=None,
        help='Optional pattern identifier'
    )
    
    parser.add_argument(
        '--generate-set',
        action='store_true',
        help='Generate complete set of patterns'
    )
    
    parser.add_argument(
        '--presets',
        type=str,
        nargs='+',
        default=None,
        help='Specific presets to generate (for --generate-set)'
    )
    
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List available pattern presets and exit'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Validate pattern size
    width, height = args.pattern
    if width < 3 or height < 3:
        raise ValueError(f"Pattern size too small: {width}x{height}. Minimum is 3x3.")
    
    if width > 20 or height > 20:
        raise ValueError(f"Pattern size too large: {width}x{height}. Maximum is 20x20.")
    
    # Validate square size
    if args.square_size <= 0:
        raise ValueError(f"Square size must be positive: {args.square_size}")
    
    if args.square_size < 5.0:
        print(f"Warning: Very small square size ({args.square_size}mm)")
    
    if args.square_size > 100.0:
        print(f"Warning: Very large square size ({args.square_size}mm)")
    
    # Generate default output filename if not provided
    if args.output is None and not args.generate_set:
        width, height = args.pattern
        args.output = f"checkerboard_{width}x{height}_{args.square_size:.0f}mm.png"


def print_generation_info(args, generator):
    """Print pattern generation information."""
    if args.generate_set:
        print("Checkerboard Pattern Set Generation")
        print("=" * 40)
        print(f"Output Directory: {args.output_dir}")
        print(f"Paper Size: {args.paper}")
        
        if args.presets:
            print(f"Presets: {', '.join(args.presets)}")
        else:
            print("Presets: All available")
    else:
        width, height = args.pattern
        print("Single Checkerboard Pattern Generation")
        print("=" * 40)
        print(f"Pattern Size: {width}x{height} internal corners")
        print(f"Square Size: {args.square_size}mm")
        print(f"Paper Size: {args.paper}")
        print(f"Output File: {args.output}")
        
        if args.pattern_id:
            print(f"Pattern ID: {args.pattern_id}")
        
        # Calculate physical dimensions
        pattern_width_mm = (width + 1) * args.square_size
        pattern_height_mm = (height + 1) * args.square_size
        print(f"Physical Size: {pattern_width_mm:.1f}x{pattern_height_mm:.1f}mm")
    
    print(f"Instructions: {'No' if args.no_instructions else 'Yes'}")
    print("=" * 40)


def generate_single_pattern(args, generator):
    """Generate a single checkerboard pattern."""
    print("Generating checkerboard pattern...")
    
    success = generator.generate_pattern(
        pattern_size=tuple(args.pattern),
        square_size_mm=args.square_size,
        output_path=args.output,
        paper_size=args.paper,
        include_instructions=not args.no_instructions,
        pattern_id=args.pattern_id
    )
    
    if success:
        print(f"\\nPattern generated successfully!")
        print(f"Output file: {args.output}")
        
        # Check if metadata file was created
        metadata_file = args.output.replace('.png', '_metadata.json')
        if os.path.exists(metadata_file):
            print(f"Metadata file: {metadata_file}")
        
        print("\\nNext steps:")
        print("1. Print the pattern at 100% scale (actual size)")
        print("2. Use high-quality printer (300+ DPI)")
        print("3. Print on matte paper to reduce reflections")
        print("4. Mount on rigid, flat surface")
        print("5. Verify dimensions with ruler before use")
        
        return True
    else:
        print("\\nPattern generation failed!")
        return False


def generate_pattern_set(args, generator):
    """Generate a complete set of patterns."""
    print("Generating checkerboard pattern set...")
    
    success = generator.generate_pattern_set(
        output_dir=args.output_dir,
        paper_size=args.paper,
        include_presets=args.presets
    )
    
    if success:
        print(f"\\nPattern set generated successfully!")
        print(f"Output directory: {args.output_dir}")
        
        # List generated files
        if os.path.exists(args.output_dir):
            pattern_files = [f for f in os.listdir(args.output_dir) if f.endswith('.png')]
            print(f"\\nGenerated {len(pattern_files)} patterns:")
            for filename in sorted(pattern_files):
                print(f"  - {filename}")
        
        summary_file = os.path.join(args.output_dir, "pattern_summary.txt")
        if os.path.exists(summary_file):
            print(f"\\nSummary file: {summary_file}")
        
        print("\\nNext steps:")
        print("1. Review the pattern_summary.txt file")
        print("2. Choose appropriate patterns for your application")
        print("3. Print selected patterns at 100% scale")
        print("4. Verify dimensions before use")
        
        return True
    else:
        print("\\nPattern set generation failed!")
        return False


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Create pattern generator
        generator = CheckerboardPatternGenerator()
        
        # Handle list presets request
        if args.list_presets:
            generator.list_presets()
            return 0
        
        # Validate arguments
        validate_arguments(args)
        
        # Print generation info
        if args.verbose:
            print_generation_info(args, generator)
            print()
        
        # Generate patterns
        if args.generate_set:
            success = generate_pattern_set(args, generator)
        else:
            success = generate_single_pattern(args, generator)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\\nPattern generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)