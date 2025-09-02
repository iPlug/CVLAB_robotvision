#!/usr/bin/env python3
"""
Checkerboard-Based Fixed Camera Calibration

This script performs high-precision camera-robot calibration using checkerboard patterns
for fixed camera setups. The robot holds and moves the checkerboard pattern while
the fixed camera observes it from different angles.

Features:
- RGB image-based checkerboard detection
- Sub-pixel corner refinement
- Camera intrinsic calibration
- Hand-to-eye transformation calculation
- Sub-5mm accuracy
- Real-time pattern detection visualization

Usage:
    python checkerboard_calibrate_camera.py [options]

Examples:
    # Standard calibration with 9x6 pattern
    python checkerboard_calibrate_camera.py
    
    # Quick calibration with 8x6 pattern
    python checkerboard_calibrate_camera.py --config checkerboard_quick --pattern 8 6
    
    # Precise calibration with custom square size
    python checkerboard_calibrate_camera.py --config checkerboard_precise --square-size 20.0
    
    # List available configurations
    python checkerboard_calibrate_camera.py --list-configs
"""

import sys
import os
import argparse
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.apps.checkerboard_calibration_app import CheckerboardCalibrationApp
from module.config.config_factory import ConfigFactory


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Checkerboard-based fixed camera calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard calibration
    python checkerboard_calibrate_camera.py
    
    # Quick calibration
    python checkerboard_calibrate_camera.py --config checkerboard_quick
    
    # Custom pattern size
    python checkerboard_calibrate_camera.py --pattern 8 6 --square-size 20.0
    
    # Verbose output
    python checkerboard_calibrate_camera.py --verbose
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='checkerboard_standard',
        help='Configuration preset to use (default: checkerboard_standard)'
    )
    
    parser.add_argument(
        '--pattern',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        default=[9, 6],
        help='Checkerboard pattern size in internal corners (default: 9 6)'
    )
    
    parser.add_argument(
        '--square-size',
        type=float,
        default=25.0,
        help='Size of checkerboard squares in millimeters (default: 25.0)'
    )
    
    parser.add_argument(
        '--output-matrix',
        type=str,
        default='camera_robot_transform.npy',
        help='Output file for transformation matrix (default: camera_robot_transform.npy)'
    )
    
    parser.add_argument(
        '--camera-calibration',
        type=str,
        default='camera_calibration.json',
        help='Camera calibration file (default: camera_calibration.json)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='checkerboard_calibration.log',
        help='Log file for calibration process (default: checkerboard_calibration.log)'
    )
    
    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available configuration presets and exit'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def list_available_configs():
    """List all available configuration presets."""
    print("Available Configuration Presets:")
    print("=" * 40)
    
    presets = ConfigFactory.get_available_presets()
    checkerboard_presets = [p for p in presets if 'checkerboard' in p or 'calibration' in p]
    
    for preset in sorted(checkerboard_presets):
        info = ConfigFactory.get_preset_info(preset)
        print(f"  {preset:25} - {info.get('description', 'No description')}")
    
    print("\\nRecommended presets for checkerboard calibration:")
    print("  checkerboard_standard  - Balanced speed and accuracy (9x6 pattern)")
    print("  checkerboard_precise   - High precision, slower (9x6 pattern, more positions)")
    print("  checkerboard_quick     - Fast calibration (8x6 pattern, fewer positions)")


def validate_pattern_size(pattern_size):
    """Validate checkerboard pattern size."""
    width, height = pattern_size
    
    if width < 3 or height < 3:
        raise ValueError(f"Pattern size too small: {width}x{height}. Minimum is 3x3.")
    
    if width > 15 or height > 15:
        raise ValueError(f"Pattern size too large: {width}x{height}. Maximum is 15x15.")
    
    # Warn about common mistakes
    if width < height:
        print(f"Warning: Pattern width ({width}) < height ({height}). "
              f"Ensure this matches your physical pattern.")


def validate_square_size(square_size):
    """Validate square size parameter."""
    if square_size <= 0:
        raise ValueError(f"Square size must be positive: {square_size}")
    
    if square_size < 5.0:
        print(f"Warning: Very small square size ({square_size}mm). "
              f"Ensure this matches your physical pattern.")
    
    if square_size > 100.0:
        print(f"Warning: Very large square size ({square_size}mm). "
              f"Ensure this matches your physical pattern.")


def print_calibration_info(args):
    """Print calibration configuration information."""
    print("Checkerboard-Based Fixed Camera Calibration")
    print("=" * 50)
    print(f"Configuration:    {args.config}")
    print(f"Pattern Size:     {args.pattern[0]}x{args.pattern[1]} internal corners")
    print(f"Square Size:      {args.square_size}mm")
    print(f"Output Matrix:    {args.output_matrix}")
    print(f"Camera Cal File:  {args.camera_calibration}")
    print(f"Log File:         {args.log_file}")
    print("=" * 50)
    
    print("\\nCalibration Process:")
    print("1. Camera intrinsic calibration (if needed)")
    print("2. Robot positioning and pattern detection")
    print("3. Hand-to-eye transformation calculation")
    print("4. Validation and accuracy assessment")
    
    print("\\nRequired Equipment:")
    print("- Intel RealSense camera (fixed position)")
    print("- myCobot robot arm") 
    print(f"- Checkerboard pattern ({args.pattern[0]}x{args.pattern[1]}, {args.square_size}mm squares)")
    print("- Good lighting conditions")


def main():
    """Main calibration function."""
    args = parse_arguments()
    
    # Handle list configs request
    if args.list_configs:
        list_available_configs()
        return 0
    
    try:
        # Validate parameters
        validate_pattern_size(args.pattern)
        validate_square_size(args.square_size)
        
        # Print configuration info
        if args.verbose:
            print_calibration_info(args)
        
        # Create calibration application
        print("\\nInitializing checkerboard calibration system...")
        
        calibration_app = CheckerboardCalibrationApp(
            config_name=args.config,
            calibration_mode='fixed_camera',
            pattern_size=tuple(args.pattern),
            square_size_mm=args.square_size,
            transformation_matrix_file=args.output_matrix,
            camera_calibration_file=args.camera_calibration,
            log_file=args.log_file
        )
        
        # Run calibration
        print("Starting calibration process...")
        print("\\nIMPORTANT INSTRUCTIONS:")
        print("1. Ensure your checkerboard pattern is securely attached to the robot")
        print("2. Check that the camera has a clear view of the workspace") 
        print("3. Ensure good, even lighting without shadows")
        print("4. Do not move the camera during calibration")
        print("\\nPress ENTER to continue...")
        input()
        
        success = calibration_app.run_full_checkerboard_calibration()
        
        if success:
            print("\\n" + "=" * 60)
            print("CALIBRATION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Transformation matrix saved to: {args.output_matrix}")
            print(f"Camera calibration saved to: {args.camera_calibration}")
            print(f"Calibration log saved to: {args.log_file}")
            print("\\nYou can now use this calibration for:")
            print("- Object tracking with robot pointing")
            print("- Pick and place operations")
            print("- Visual servoing applications")
            return 0
        else:
            print("\\n" + "=" * 60)
            print("CALIBRATION FAILED!")
            print("=" * 60)
            print("Please check the log file for detailed error information:")
            print(f"  {args.log_file}")
            print("\\nCommon issues:")
            print("- Poor lighting or shadows on checkerboard")
            print("- Camera or robot connection problems")
            print("- Insufficient pattern detections")
            print("- Robot positioning errors")
            return 1
            
    except KeyboardInterrupt:
        print("\\n\\nCalibration interrupted by user")
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