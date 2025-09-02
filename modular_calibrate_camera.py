#!/usr/bin/env python3
"""
Modular Camera-Robot Calibration Script

This script provides a modular implementation of camera-robot calibration
using the enhanced modular architecture. It supports multiple calibration modes,
async robot control, and real-time visualization feedback.

Usage:
    python modular_calibrate_camera.py [--config CONFIG] [--output OUTPUT] [--log LOG]

Examples:
    # Standard calibration (5 points, balanced speed/accuracy)
    python modular_calibrate_camera.py

    # High-precision calibration (7 points, slow and accurate)
    python modular_calibrate_camera.py --config calibration_precise

    # Quick calibration (3 points, fast)
    python modular_calibrate_camera.py --config calibration_quick

    # Custom output file
    python modular_calibrate_camera.py --output my_transform.npy --log my_calibration.log

Features compared to original calibrate_camera.py:
    - Modular architecture using existing components
    - Async robot movement for smooth UI operation
    - Real-time 3-panel visualization (Color | BEV | Point Cloud)
    - Multiple calibration presets
    - Enhanced object detection with outlier filtering
    - Improved error handling and logging
    - Configuration-based calibration positions
"""

import argparse
import sys
import os

# Add module path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from module.apps.calibration_app import ModularCalibrationApp
from module.config.config_factory import ConfigFactory


def print_available_configs():
    """Print available calibration configurations."""
    configs = ['calibration_standard', 'calibration_precise', 'calibration_quick', 'virtual_fixed_camera']
    
    print("\\nAvailable Calibration Configurations:")
    print("=" * 50)
    
    for config_name in configs:
        try:
            config = ConfigFactory.create_preset(config_name)
            positions = config.get('calibration_positions', [])
            thresholds = config.get('accuracy_thresholds', {})
            robot_params = config.get('robot_params', {})
            
            print(f"\\n{config_name}:")
            print(f"  Points: {len(positions)}")
            print(f"  Speed: {robot_params.get('max_speed', 'N/A')}")
            print(f"  Max Error: {thresholds.get('max_calibration_error', 'N/A')}mm")
            
            if config_name == 'calibration_standard':
                print("  Description: Balanced speed and accuracy (recommended)")
            elif config_name == 'calibration_precise':
                print("  Description: High precision, slower calibration")
            elif config_name == 'calibration_quick':
                print("  Description: Fast calibration, lower precision")
            elif config_name == 'virtual_fixed_camera':
                print("  Description: End-effector camera acting as virtual fixed camera")
                
        except Exception as e:
            print(f"  Error loading {config_name}: {e}")


def validate_config(config_name: str) -> bool:
    """Validate that the specified configuration exists and is valid."""
    try:
        config = ConfigFactory.create_preset(config_name)
        
        # Check required fields using config.get() method
        required_fields = ['calibration_positions', 'robot_params', 'accuracy_thresholds']
        for field in required_fields:
            field_value = config.get(field)
            if field_value is None:
                print(f"Error: Configuration '{config_name}' missing required field: {field}")
                return False
        
        # Validate calibration positions
        positions = config.get('calibration_positions', [])
        if len(positions) < 3:
            print(f"Error: Configuration '{config_name}' has too few calibration positions: {len(positions)}")
            return False
        
        for i, pos in enumerate(positions):
            if len(pos) < 6:
                print(f"Error: Position {i+1} in '{config_name}' should have 6 elements [x,y,z,rx,ry,rz]")
                return False
        
        print(f"[OK] Configuration '{config_name}' is valid")
        return True
        
    except Exception as e:
        print(f"Error: Failed to load configuration '{config_name}': {e}")
        return False


def print_calibration_info(config_name: str):
    """Print detailed information about the calibration process."""
    try:
        config = ConfigFactory.create_preset(config_name)
        positions = config.get('calibration_positions', [])
        test_positions = config.get('test_positions', [])
        thresholds = config.get('accuracy_thresholds', {})
        robot_params = config.get('robot_params', {})
        
        print(f"\\nCalibration Configuration: {config_name}")
        print("=" * 60)
        print(f"Calibration Points: {len(positions)}")
        print(f"Validation Points: {len(test_positions)}")
        print(f"Robot Speed: {robot_params.get('max_speed', 'N/A')}")
        print(f"Max Calibration Error: {thresholds.get('max_calibration_error', 'N/A')}mm")
        print(f"Max Validation Error: {thresholds.get('max_validation_error', 'N/A')}mm")
        
        print(f"\\nCalibration Positions:")
        for i, pos in enumerate(positions):
            print(f"  {i+1}. [{pos[0]}, {pos[1]}, {pos[2]}, {pos[3]}, {pos[4]}, {pos[5]}]")
        
        if test_positions:
            print(f"\\nValidation Positions:")
            for i, pos in enumerate(test_positions):
                print(f"  {i+1}. [{pos[0]}, {pos[1]}, {pos[2]}, {pos[3]}, {pos[4]}, {pos[5]}]")
        
        print("\\nCalibration Process:")
        print("1. Robot moves to each calibration position")
        print("2. User places object below robot end effector")
        print("3. Robot moves to home position")
        print("4. Camera detects object position")
        print("5. Transformation matrix calculated from all points")
        print("6. Validation tests verify accuracy")
        print("7. Matrix saved to output file")
        
    except Exception as e:
        print(f"Error loading configuration info: {e}")


def main():
    """Main entry point for modular camera calibration."""
    parser = argparse.ArgumentParser(
        description='Modular Camera-Robot Calibration System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Standard calibration
  %(prog)s --config calibration_precise      # High-precision calibration
  %(prog)s --config calibration_quick        # Quick calibration
  %(prog)s --output my_transform.npy         # Custom output file
  %(prog)s --list-configs                    # Show available configurations
  %(prog)s --info calibration_precise       # Show config details
        """
    )
    
    parser.add_argument('--config', default='calibration_standard',
                       help='Calibration configuration preset (default: calibration_standard)')
    parser.add_argument('--output', default='camera_robot_transform.npy',
                       help='Output file for transformation matrix (default: camera_robot_transform.npy)')
    parser.add_argument('--log', default='calibration_report.log',
                       help='Log file for calibration process (default: calibration_report.log)')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available calibration configurations and exit')
    parser.add_argument('--info', metavar='CONFIG',
                       help='Show detailed information about a configuration and exit')
    parser.add_argument('--validate-config', action='store_true',
                       help='Validate the specified configuration and exit')
    parser.add_argument('--disable-auto-roi', action='store_true',
                       help='Disable automatic ROI detection based on table surface')
    parser.add_argument('--floor-offset', type=float, default=0.01,
                       help='Height offset above detected table surface in meters (default: 0.01m = 10mm)')
    
    args = parser.parse_args()
    
    # Handle info commands
    if args.list_configs:
        print_available_configs()
        return 0
    
    if args.info:
        print_calibration_info(args.info)
        return 0
    
    # Validate configuration
    if not validate_config(args.config):
        print(f"\\nUse --list-configs to see available configurations")
        return 1
    
    if args.validate_config:
        print(f"Configuration '{args.config}' is valid")
        return 0
    
    # Print startup information
    print("Modular Camera-Robot Calibration System")
    print("=" * 50)
    print(f"Configuration: {args.config}")
    print(f"Output file: {args.output}")
    print(f"Log file: {args.log}")
    print(f"Auto ROI detection: {'Disabled' if args.disable_auto_roi else 'Enabled'}")
    if not args.disable_auto_roi:
        print(f"Floor offset: {args.floor_offset}m")
    print("=" * 50)
    
    # Show configuration details
    print_calibration_info(args.config)
    
    # Confirm before starting
    print("\\nIMPORTANT SAFETY NOTES:")
    print("- Ensure robot workspace is clear of obstacles")
    print("- Have emergency stop readily accessible")
    print("- Camera and robot must be in fixed positions during calibration")
    print("- Use a distinctive calibration object (colored block, etc.)")
    
    confirm = input("\\nReady to start calibration? (y/n): ").lower().strip()
    if confirm != 'y':
        print("Calibration cancelled")
        return 0
    
    try:
        # Create and run calibration app
        calibration_app = ModularCalibrationApp(
            config_name=args.config,
            transformation_matrix_file=args.output,
            log_file=args.log,
            enable_auto_roi=not args.disable_auto_roi,
            floor_offset=args.floor_offset
        )
        
        success = calibration_app.run_full_calibration()
        
        if success:
            print("\\n🎉 Calibration completed successfully!")
            print(f"Transformation matrix saved to: {args.output}")
            print(f"Calibration log saved to: {args.log}")
            print("\\nYou can now use the modular tracking system:")
            print(f"  python modular_track_object.py --config realtime_tracking")
            return 0
        else:
            print("\\n❌ Calibration failed!")
            print("Check the log file for details and try again.")
            return 1
            
    except KeyboardInterrupt:
        print("\\n\\nCalibration interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n\\nUnexpected error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())