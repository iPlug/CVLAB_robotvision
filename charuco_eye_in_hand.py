#!/usr/bin/env python3
"""
CharUco-Based Eye-in-Hand Calibration

This script performs high-precision camera-robot calibration using CharUco patterns
for eye-in-hand setups. The robot-mounted camera observes a fixed CharUco pattern
from different positions and orientations.

Features:
- RGB image-based CharUco detection
- ArUco marker and ChArUco corner detection
- Sub-pixel corner refinement
- Camera intrinsic calibration
- Eye-in-hand transformation calculation
- Sub-5mm accuracy
- Real-time pattern detection visualization

Usage:
    python charuco_eye_in_hand.py [options]

Examples:
    # Standard eye-in-hand calibration with 8x11 CharUco
    python charuco_eye_in_hand.py
    
    # Quick calibration with custom marker size
    python charuco_eye_in_hand.py --config charuco_quick --marker-size 18.0
    
    # Precise calibration with custom square size
    python charuco_eye_in_hand.py --config charuco_precise --square-size 30.0
    
    # Custom pattern position on table
    python charuco_eye_in_hand.py --pattern-position 300 0 0
"""

import sys
import os
import argparse
import time

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.apps.charuco_calibration_app import CharucoCalibrationApp
from module.config.config_factory import ConfigFactory


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CharUco-based eye-in-hand calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard eye-in-hand calibration
    python charuco_eye_in_hand.py
    
    # Quick calibration
    python charuco_eye_in_hand.py --config charuco_quick
    
    # Custom pattern and position
    python charuco_eye_in_hand.py --pattern 8 11 --pattern-position 350 0 0
    
    # Verbose output with custom marker size
    python charuco_eye_in_hand.py --marker-size 22.0 --verbose
        """
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='charuco_standard',
        help='Configuration preset to use (default: charuco_standard)'
    )
    
    parser.add_argument(
        '--pattern',
        type=int,
        nargs=2,
        metavar=('WIDTH', 'HEIGHT'),
        default=[8, 11],
        help='CharUco pattern size in squares (default: 8 11)'
    )
    
    parser.add_argument(
        '--square-size',
        type=float,
        default=25.0,
        help='Size of checkerboard squares in millimeters (default: 25.0)'
    )
    
    parser.add_argument(
        '--marker-size',
        type=float,
        default=20.0,
        help='Size of ArUco markers in millimeters (default: 20.0)'
    )
    
    parser.add_argument(
        '--pattern-position',
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        default=[400, 0, 0],
        help='Fixed pattern position on table [x, y, z] in mm (default: 400 0 0)'
    )
    
    parser.add_argument(
        '--output-matrix',
        type=str,
        default='charuco_eye_in_hand_transform.npy',
        help='Output file for transformation matrix (default: charuco_eye_in_hand_transform.npy)'
    )
    
    parser.add_argument(
        '--camera-calibration',
        type=str,
        default='charuco_camera_calibration.json',
        help='Camera calibration file (default: charuco_camera_calibration.json)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        default='charuco_eye_in_hand_calibration.log',
        help='Log file for calibration process (default: charuco_eye_in_hand_calibration.log)'
    )
    
    parser.add_argument(
        '--num-positions',
        type=int,
        default=12,
        help='Number of calibration positions (default: 12)'
    )
    
    parser.add_argument(
        '--hemisphere-coverage',
        action='store_true',
        help='Use hemisphere coverage pattern for camera positions'
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
    
    parser.add_argument(
        '--no-filtering',
        action='store_true',
        help='Disable quality filtering - use permissive detection like test_charuco_detection.py'
    )
    
    return parser.parse_args()


def list_available_configs():
    """List all available configuration presets."""
    print("Available Configuration Presets:")
    print("=" * 40)
    
    presets = ConfigFactory.get_available_presets()
    charuco_presets = [p for p in presets if 'charuco' in p or 'calibration' in p]
    
    for preset in sorted(charuco_presets):
        info = ConfigFactory.get_preset_info(preset)
        print(f"  {preset:25} - {info.get('description', 'No description')}")
    
    print("\\nRecommended presets for eye-in-hand calibration:")
    print("  charuco_standard       - Balanced speed and accuracy (12 positions)")
    print("  charuco_precise        - High precision, more positions (15+ positions)")
    print("  charuco_quick          - Fast calibration (8 positions)")


def validate_pattern_position(position):
    """Validate pattern position parameters."""
    x, y, z = position
    
    # Check reasonable workspace bounds
    if abs(x) > 600:
        print(f"Warning: X position ({x}mm) is quite far from robot base")
    
    if abs(y) > 400:
        print(f"Warning: Y position ({y}mm) is quite far from robot center")
    
    if z < -50 or z > 100:
        print(f"Warning: Z position ({z}mm) might be outside safe range (-50 to 100mm)")
    
    # Check if position is reachable
    distance = (x**2 + y**2 + z**2)**0.5
    if distance > 350:  # myCobot 280 reach is about 280mm
        print(f"Warning: Pattern position distance ({distance:.0f}mm) may be outside robot reach")


def print_calibration_info(args):
    """Print calibration configuration information."""
    print("CharUco-Based Eye-in-Hand Calibration")
    print("=" * 50)
    print(f"Configuration:     {args.config}")
    print(f"Pattern Size:      {args.pattern[0]}x{args.pattern[1]} squares")
    print(f"Square Size:       {args.square_size}mm")
    print(f"Marker Size:       {args.marker_size}mm")
    print(f"Pattern Position:  [{args.pattern_position[0]}, {args.pattern_position[1]}, {args.pattern_position[2]}] mm")
    print(f"Num Positions:     {args.num_positions}")
    print(f"Coverage Pattern:  {'Hemisphere' if args.hemisphere_coverage else 'Grid'}")
    print(f"Output Matrix:     {args.output_matrix}")
    print(f"Camera Cal File:   {args.camera_calibration}")
    print(f"Log File:          {args.log_file}")
    print("=" * 50)
    
    print("\\nCalibration Process:")
    print("1. Camera intrinsic calibration (if needed)")
    print("2. Place CharUco pattern at fixed position")
    print("3. Robot moves camera to multiple viewing positions")
    print("4. Eye-in-hand transformation calculation")
    print("5. Validation and accuracy assessment")
    
    print("\\nRequired Equipment:")
    print("- Intel RealSense camera (robot-mounted)")
    print("- myCobot robot arm")
    print(f"- CharUco pattern ({args.pattern[0]}x{args.pattern[1]}, {args.square_size}mm squares, {args.marker_size}mm markers)")
    print("- Stable table surface")
    print("- Good lighting conditions")


def create_custom_config(args):
    """Create custom configuration based on command line arguments."""
    # Load base configuration
    base_config = ConfigFactory.create_preset(args.config)
    
    # Update with custom parameters
    if hasattr(base_config, 'config'):
        # Update pattern position for eye-in-hand mode
        if 'calibration_modes' in base_config.config:
            if 'eye_in_hand' in base_config.config['calibration_modes']:
                base_config.config['calibration_modes']['eye_in_hand']['fixed_pattern_position'] = args.pattern_position
                base_config.config['calibration_modes']['eye_in_hand']['num_positions'] = args.num_positions
        
        # Update pattern parameters
        if 'pattern_params' in base_config.config:
            base_config.config['pattern_params']['pattern_size'] = tuple(args.pattern)
            base_config.config['pattern_params']['square_size_mm'] = args.square_size
            base_config.config['pattern_params']['marker_size_mm'] = args.marker_size
    
    return base_config


def main():
    """Main calibration function."""
    args = parse_arguments()
    
    # Handle list configs request
    if args.list_configs:
        list_available_configs()
        return 0
    
    try:
        # Validate parameters
        validate_pattern_position(args.pattern_position)
        
        # Print configuration info
        if args.verbose:
            print_calibration_info(args)
        
        # Create calibration application
        print("\\nInitializing eye-in-hand CharUco calibration system...")
        
        calibration_app = CharucoCalibrationApp(
            config_name=args.config,
            calibration_mode='eye_in_hand',
            pattern_size=tuple(args.pattern),
            square_size_mm=args.square_size,
            marker_size_mm=args.marker_size,
            transformation_matrix_file=args.output_matrix,
            camera_calibration_file=args.camera_calibration,
            log_file=args.log_file,
            no_filtering=args.no_filtering
        )
        
        # Update configuration with custom pattern position
        if hasattr(calibration_app.config, 'config'):
            if 'calibration_modes' in calibration_app.config.config:
                if 'eye_in_hand' in calibration_app.config.config['calibration_modes']:
                    print(f"DEBUG: Overriding pattern position from {calibration_app.config.config['calibration_modes']['eye_in_hand']['fixed_pattern_position']} to {args.pattern_position}")
                    calibration_app.config.config['calibration_modes']['eye_in_hand']['fixed_pattern_position'] = args.pattern_position
                    calibration_app.config.config['calibration_modes']['eye_in_hand']['num_positions'] = args.num_positions
        
        # Run calibration
        print("Starting eye-in-hand calibration process...")
        print("\\nIMPORTANT INSTRUCTIONS:")
        print("1. Ensure the RealSense camera is securely mounted on the robot")
        print("2. Place the CharUco pattern on the table at the specified position")
        print(f"   Pattern position: [{args.pattern_position[0]}, {args.pattern_position[1]}, {args.pattern_position[2]}] mm")
        print("3. Ensure the pattern will remain stationary throughout calibration")
        print("4. Check that lighting is good and even (no shadows on pattern)")
        print("5. Ensure the robot workspace is clear for camera movement")
        print("\\nThe robot will move the camera to multiple positions to view the pattern.")
        print("\\nStarting calibration automatically...")
        # input()  # Removed for automatic operation
        
        success = calibration_app.run_full_charuco_calibration()
        
        if success:
            print("\\n" + "=" * 60)
            print("CHARUCO EYE-IN-HAND CALIBRATION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Transformation matrix saved to: {args.output_matrix}")
            print(f"Camera calibration saved to: {args.camera_calibration}")
            print(f"Calibration log saved to: {args.log_file}")
            print("\\nYou can now use this calibration for:")
            print("- Eye-in-hand visual servoing with CharUco patterns")
            print("- High-precision robot-guided object inspection")
            print("- Dynamic pick and place operations")
            print("- Real-time visual feedback control")
            return 0
        else:
            print("\\n" + "=" * 60)
            print("CALIBRATION FAILED!")
            print("=" * 60)
            print("Please check the log file for detailed error information:")
            print(f"  {args.log_file}")
            print("\\nCommon issues:")
            print("- Poor lighting or shadows on CharUco pattern")
            print("- Pattern not visible from all camera positions")
            print("- Camera or robot connection problems")
            print("- Insufficient CharUco corner detections")
            print("- Robot positioning errors")
            print("- ArUco markers not clearly visible")
            print("\\nTroubleshooting tips:")
            print("- Ensure CharUco pattern is printed clearly and at correct size")
            print("- Improve lighting (avoid shadows and reflections)")
            print("- Check that pattern position is within robot workspace")
            print("- Verify camera is securely mounted and calibrated")
            print("- Ensure ArUco markers are sharp and not damaged")
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