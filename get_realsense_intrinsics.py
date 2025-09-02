#!/usr/bin/env python3
"""
Extract and display RealSense camera intrinsics from the SDK.
Shows how to get factory intrinsics for any RealSense camera.
"""

import pyrealsense2 as rs
import numpy as np
import json

def get_realsense_intrinsics():
    """Get RealSense camera intrinsics from the SDK."""
    try:
        # Initialize RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Try common resolutions
        resolutions = [
            (1280, 720, 30),  # HD
            (960, 540, 30),   # Mid
            (640, 480, 30),   # VGA
        ]
        
        intrinsics_data = {}
        
        for width, height, fps in resolutions:
            try:
                print(f"\n=== TESTING RESOLUTION: {width}x{height}@{fps}fps ===")
                
                # Configure stream
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                
                # Start pipeline
                profile = pipeline.start(config)
                
                # Get color stream profile
                color_profile = profile.get_stream(rs.stream.color)
                color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
                
                # Get device info
                device = profile.get_device()
                device_name = device.get_info(rs.camera_info.name)
                serial_number = device.get_info(rs.camera_info.serial_number)
                firmware_version = device.get_info(rs.camera_info.firmware_version)
                
                print(f"Device: {device_name}")
                print(f"Serial: {serial_number}")
                print(f"Firmware: {firmware_version}")
                print(f"Resolution: {width}x{height}")
                print()
                
                # Display intrinsics
                print("CAMERA INTRINSICS:")
                print(f"  fx = {color_intrinsics.fx:.6f}")
                print(f"  fy = {color_intrinsics.fy:.6f}")
                print(f"  ppx (cx) = {color_intrinsics.ppx:.6f}")
                print(f"  ppy (cy) = {color_intrinsics.ppy:.6f}")
                print(f"  Distortion model: {color_intrinsics.model}")
                print(f"  Distortion coeffs: {color_intrinsics.coeffs}")
                
                # Create OpenCV camera matrix
                camera_matrix = np.array([
                    [color_intrinsics.fx, 0.0, color_intrinsics.ppx],
                    [0.0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0.0, 0.0, 1.0]
                ])
                
                print(f"OpenCV Camera Matrix:")
                print(camera_matrix)
                print()
                
                # Store data for this resolution
                resolution_key = f"{width}x{height}@{fps}"
                intrinsics_data[resolution_key] = {
                    'device_info': {
                        'name': device_name,
                        'serial_number': serial_number,
                        'firmware_version': firmware_version
                    },
                    'resolution': {
                        'width': width,
                        'height': height,
                        'fps': fps
                    },
                    'intrinsics': {
                        'fx': color_intrinsics.fx,
                        'fy': color_intrinsics.fy,
                        'ppx': color_intrinsics.ppx,
                        'ppy': color_intrinsics.ppy,
                        'model': str(color_intrinsics.model),
                        'coeffs': list(color_intrinsics.coeffs)
                    },
                    'camera_matrix': camera_matrix.tolist(),
                    'dist_coeffs': list(color_intrinsics.coeffs)
                }
                
                # Stop pipeline for this resolution
                pipeline.stop()
                config.disable_all_streams()
                
            except Exception as e:
                print(f"Failed to get intrinsics for {width}x{height}: {e}")
                try:
                    pipeline.stop()
                except:
                    pass
                config.disable_all_streams()
                continue
        
        # Save all intrinsics data
        if intrinsics_data:
            output_file = 'realsense_intrinsics_all_resolutions.json'
            with open(output_file, 'w') as f:
                json.dump(intrinsics_data, f, indent=2)
            print(f"All intrinsics data saved to: {output_file}")
            
            # Show how to use in code
            print("\n" + "="*60)
            print("HOW TO USE REALSENSE INTRINSICS IN YOUR CODE:")
            print("="*60)
            
            # Get the first successful resolution as example
            example_res = next(iter(intrinsics_data.keys()))
            example_data = intrinsics_data[example_res]
            
            print("# Method 1: Direct from SDK (runtime)")
            print("import pyrealsense2 as rs")
            print("pipeline = rs.pipeline()")
            print("profile = pipeline.start()")
            print("color_profile = profile.get_stream(rs.stream.color)")
            print("intrinsics = color_profile.as_video_stream_profile().get_intrinsics()")
            print("camera_matrix = np.array([")
            print("    [intrinsics.fx, 0.0, intrinsics.ppx],")
            print("    [0.0, intrinsics.fy, intrinsics.ppy],")
            print("    [0.0, 0.0, 1.0]")
            print("])")
            print()
            
            print("# Method 2: Hardcoded values (faster startup)")
            print("camera_matrix = np.array([")
            for row in example_data['camera_matrix']:
                print(f"    {row},")
            print("])")
            print(f"dist_coeffs = np.array({example_data['dist_coeffs']})")
            print()
            
            print("KEY DIFFERENCES FROM CALIBRATION:")
            print("- RealSense SDK: Factory calibration, unknown reference distance")
            print("- Custom calibration: Optimized for your specific working distance")
            print("- RealSense SDK: Fixed for all applications")  
            print("- Custom calibration: Tuned for CharUco pattern detection")
            
        else:
            print("No intrinsics data collected - camera may not be connected")
            
    except Exception as e:
        print(f"Error getting RealSense intrinsics: {e}")

if __name__ == "__main__":
    get_realsense_intrinsics()