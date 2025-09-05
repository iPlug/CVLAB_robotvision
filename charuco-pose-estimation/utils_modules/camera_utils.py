#!/usr/bin/env python3
"""
Camera utilities for CharUco pose estimation applications.
Handles camera setup, intrinsics management, and calibration loading.
"""

import sys
import json
import numpy as np
import pyrealsense2 as rs
from pathlib import Path


class CameraUtils:
    """Utilities for camera setup, intrinsics management, and eye-in-hand calibration loading."""
    
    def __init__(self, config_path="charuco-pose-estimation"):
        """Initialize camera utilities."""
        self.config_path = Path(config_path)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros(4)
        self.color_intrinsics = None
        
    def load_camera_intrinsics(self):
        """Load camera intrinsics from file."""
        intrinsic_file = self.config_path / 'intrinsic_l515.txt'
        if intrinsic_file.exists():
            try:
                self.camera_matrix = np.loadtxt(intrinsic_file)
                print(f"Loaded camera intrinsics from {intrinsic_file}")
                print(f"Camera matrix:")
                print(f"  fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")
                print(f"  cx={self.camera_matrix[0,2]:.2f}, cy={self.camera_matrix[1,2]:.2f}")
                print(f"Matrix:\n{self.camera_matrix}")
                self.dist_coeffs = np.zeros(5)
                print(f"Distortion coefficients: {self.dist_coeffs}")
            except Exception as e:
                print(f"Warning: Could not load camera intrinsics: {e}")
                print("Using default camera parameters")
                self.camera_matrix = np.array([
                    [615.0, 0.0, 320.0],
                    [0.0, 615.0, 240.0],
                    [0.0, 0.0, 1.0]
                ])
                self.dist_coeffs = np.zeros(5)
        else:
            print(f"No intrinsics file found at {intrinsic_file}")
            print("Using default camera parameters")
            self.camera_matrix = np.array([
                [615.0, 0.0, 320.0],
                [0.0, 615.0, 240.0],
                [0.0, 0.0, 1.0]
            ])
            self.dist_coeffs = np.zeros(5)
    
    def setup_camera(self, pipeline, config):
        """Setup RealSense camera with L515 optimized settings."""
        try:
            color_depth_configs = [
                ((640, 480, 30), (640, 480, 30)),
                ((1280, 720, 30), (1280, 720, 30)),
            ]
            
            profile = None
            for color_config, depth_config in color_depth_configs:
                color_w, color_h, color_fps = color_config
                depth_w, depth_h, depth_fps = depth_config
                try:
                    print(f"Trying color: {color_w}x{color_h}@{color_fps}fps, depth: {depth_w}x{depth_h}@{depth_fps}fps")
                    config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, color_fps)
                    config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, depth_fps)
                    profile = pipeline.start(config)
                    
                    try:
                        device = profile.get_device()
                        depth_sensor = device.first_depth_sensor()
                        if hasattr(rs, 'l500_visual_preset') and hasattr(depth_sensor, 'set_option'):
                            depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.short_range)
                            print("✅ Applied L515 Short Range Preset (0.25m-1.5m optimal)")
                        else:
                            print("⚠️  L515 visual presets not available, using default settings")
                    except Exception as preset_error:
                        print(f"⚠️  Could not apply L515 preset: {preset_error}")
                    
                    print(f"Successfully initialized - Color: {color_w}x{color_h}@{color_fps}fps, Depth: {depth_w}x{depth_h}@{depth_fps}fps")
                    break
                except Exception as e:
                    print(f"Failed color:{color_w}x{color_h} depth:{depth_w}x{depth_h}: {e}")
                    try:
                        pipeline.stop()
                    except:
                        pass
                    config.disable_all_streams()
                    continue
            
            if profile is None:
                raise Exception("No supported resolution found")
            
            color_profile = profile.get_stream(rs.stream.color)
            intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            self.color_intrinsics = intrinsics
            
            color_width, color_height = intrinsics.width, intrinsics.height
            calibration_width, calibration_height = 1280, 720
            
            if (color_width, color_height) == (calibration_width, calibration_height):
                print("✅ Perfect match! Using original calibration resolution")
            else:
                print(f"⚠️  Resolution mismatch: Current {color_width}x{color_height}, Calibration {calibration_width}x{calibration_height}")
                scale_x = color_width / calibration_width
                scale_y = color_height / calibration_height
                
                if hasattr(self, 'camera_matrix') and self.camera_matrix is not None:
                    original_fx = self.camera_matrix[0, 0]
                    original_fy = self.camera_matrix[1, 1] 
                    original_cx = self.camera_matrix[0, 2]
                    original_cy = self.camera_matrix[1, 2]
                    
                    scaled_fx = original_fx * scale_x
                    scaled_fy = original_fy * scale_y
                    scaled_cx = original_cx * scale_x
                    scaled_cy = original_cy * scale_y
                    
                    self.camera_matrix = np.array([
                        [scaled_fx, 0.0, scaled_cx],
                        [0.0, scaled_fy, scaled_cy],
                        [0.0, 0.0, 1.0]
                    ])
                    
                    print(f"Scaled calibration intrinsics:")
                    print(f"  fx={scaled_fx:.2f} (was {original_fx:.2f})")
                    print(f"  fy={scaled_fy:.2f} (was {original_fy:.2f})")
                    print(f"  cx={scaled_cx:.2f} (was {original_cx:.2f})")
                    print(f"  cy={scaled_cy:.2f} (was {original_cy:.2f})")
                else:
                    print("No calibration intrinsics found - using RealSense intrinsics")
                    self.camera_matrix = np.array([
                        [intrinsics.fx, 0.0, intrinsics.ppx],
                        [0.0, intrinsics.fy, intrinsics.ppy],
                        [0.0, 0.0, 1.0]
                    ])
            
            self.dist_coeffs = np.array(intrinsics.coeffs[:5])
            
            device = profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            serial_number = device.get_info(rs.camera_info.serial_number)
            firmware_version = device.get_info(rs.camera_info.firmware_version)
            
            print("RealSense camera initialized successfully")
            print(f"Device: {device_name} (S/N: {serial_number}, FW: {firmware_version})")
            print(f"Resolution: {intrinsics.width}x{intrinsics.height}")
            print(f"Camera intrinsics from RealSense SDK:")
            print(f"  fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
            print(f"  cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")
            print(f"Camera matrix:\n{self.camera_matrix}")
            print(f"Distortion coefficients: {self.dist_coeffs}")
            
            try:
                intrinsics_file = 'intrinsic_from_realsense.txt'
                np.savetxt(intrinsics_file, self.camera_matrix)
                print(f"Intrinsics saved to: {intrinsics_file}")
            except Exception as e:
                print(f"Could not save intrinsics file: {e}")
            
            calibration_data = {
                'camera_matrix': self.camera_matrix.tolist(),
                'distortion_coefficients': self.dist_coeffs.tolist(),
                'device_info': {
                    'name': device_name,
                    'serial_number': serial_number,
                    'firmware_version': firmware_version
                },
                'intrinsics': {
                    'fx': intrinsics.fx,
                    'fy': intrinsics.fy,
                    'ppx': intrinsics.ppx,
                    'ppy': intrinsics.ppy,
                    'width': intrinsics.width,
                    'height': intrinsics.height,
                    'model': str(intrinsics.model)
                }
            }
            
            try:
                with open('camera_calibration_realsense.json', 'w') as f:
                    json.dump(calibration_data, f, indent=2)
                print(f"Detailed calibration saved to: camera_calibration_realsense.json")
            except Exception as e:
                print(f"Could not save calibration data: {e}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize RealSense camera: {e}")
            return False
    
    def load_eye_in_hand_transform(self):
        """Load pre-calibrated eye-in-hand transformation matrix and its intrinsics."""
        calibration_dir = self.config_path / 'calibration_result_1754622347'
        transform_file = calibration_dir / 'eye_in_hand_transform.npy'
        intrinsics_file = calibration_dir / 'camera_intrinsics.txt'
        calibration_json = calibration_dir / 'calibration_data.json'
        
        try:
            transform = np.load(transform_file)
            print(f"Loaded eye-in-hand transformation from: {transform_file}")
            print("Transformation matrix:")
            for row in transform:
                print(f"  [{row[0]:8.5f} {row[1]:8.5f} {row[2]:8.5f} {row[3]:8.5f}]")
            
            if intrinsics_file.exists():
                calibration_intrinsics = np.loadtxt(intrinsics_file)
                print(f"\nFound calibration intrinsics: {intrinsics_file}")
                print("Calibration used these intrinsics:")
                print(f"  fx={calibration_intrinsics[0,0]:.2f}, fy={calibration_intrinsics[1,1]:.2f}")
                print(f"  cx={calibration_intrinsics[0,2]:.2f}, cy={calibration_intrinsics[1,2]:.2f}")
                self.camera_matrix = calibration_intrinsics
                self.dist_coeffs = np.zeros(5)
                print("USING CALIBRATION INTRINSICS (recommended)")
                
            elif calibration_json.exists():
                try:
                    with open(calibration_json, 'r') as f:
                        calib_data = json.load(f)
                    if 'camera_intrinsics' in calib_data:
                        calibration_intrinsics = np.array(calib_data['camera_intrinsics']['camera_matrix'])
                        print(f"\nFound calibration intrinsics in JSON: {calibration_json}")
                        print("Calibration used these intrinsics:")
                        print(f"  fx={calibration_intrinsics[0,0]:.2f}, fy={calibration_intrinsics[1,1]:.2f}")
                        print(f"  cx={calibration_intrinsics[0,2]:.2f}, cy={calibration_intrinsics[1,2]:.2f}")
                        self.camera_matrix = calibration_intrinsics
                        self.dist_coeffs = np.array(calib_data['camera_intrinsics']['dist_coeffs'])
                        print("USING CALIBRATION INTRINSICS (recommended)")
                except Exception as e:
                    print(f"Could not load intrinsics from JSON: {e}")
            else:
                print(f"⚠️  WARNING: No calibration intrinsics found!")
                print(f"   Looked for: {intrinsics_file}")  
                print(f"   This transformation was calibrated with different intrinsics")
                print(f"   Using currently loaded intrinsics may give wrong results")
            
            return transform
            
        except FileNotFoundError:
            print(f"Error: Eye-in-hand transformation not found at: {transform_file}")
            print("Please run the calibration first using charuco_realtime_pose.py")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading eye-in-hand transformation: {e}")
            sys.exit(1)