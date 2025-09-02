#!/usr/bin/env python3
"""
Simple RealSense L515 Checkerboard Detection Test
Detects 7x5 checkerboard pattern and shows results in real-time.
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time

def main():
    """Main detection loop."""
    print("RealSense L515 Checkerboard Detection Test")
    print("Pattern: 7x5 (7 corners wide, 5 corners tall)")
    print("Press 'q' to quit, 's' to save image, 'c' to capture detection")
    print("=" * 50)
    
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream (RGB)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # Start streaming
        print("Starting RealSense L515 camera...")
        profile = pipeline.start(config)
        
        # Get camera intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        print(f"Camera intrinsics: fx={color_intrinsics.fx:.1f}, fy={color_intrinsics.fy:.1f}")
        print(f"Principal point: cx={color_intrinsics.ppx:.1f}, cy={color_intrinsics.ppy:.1f}")
        
        # Checkerboard parameters
        pattern_size = (7, 5)  # 7 corners wide, 5 corners tall
        square_size_mm = 15.0  # 15mm squares
        
        # Detection criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Statistics
        detection_count = 0
        total_attempts = 0
        
        print("Camera ready! Looking for 7x5 checkerboard pattern...")
        
        while True:
            # Get frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert BGR to RGB for processing
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            
            total_attempts += 1
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray, 
                pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )
            
            # Create display image
            display_image = color_image.copy()
            
            if ret:
                detection_count += 1
                
                # Refine corners
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Draw corners
                cv2.drawChessboardCorners(display_image, pattern_size, corners_refined, ret)
                
                # Calculate pattern center
                center = np.mean(corners_refined, axis=0)[0]
                cv2.circle(display_image, tuple(center.astype(int)), 5, (0, 255, 0), -1)
                
                # Add detection info
                cv2.putText(display_image, f"DETECTED: 7x5 Pattern", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_image, f"Corners: {len(corners_refined)}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_image, f"Center: ({center[0]:.1f}, {center[1]:.1f})", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Calculate detection quality (simple metric)
                # Check if corners are well distributed
                min_x, min_y = np.min(corners_refined.reshape(-1, 2), axis=0)
                max_x, max_y = np.max(corners_refined.reshape(-1, 2), axis=0)
                pattern_width = max_x - min_x
                pattern_height = max_y - min_y
                pattern_area = pattern_width * pattern_height
                image_area = gray.shape[0] * gray.shape[1]
                coverage = pattern_area / image_area
                
                cv2.putText(display_image, f"Coverage: {coverage:.1%}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                print(f"\\rDetection {detection_count}: Center=({center[0]:.1f}, {center[1]:.1f}), Coverage={coverage:.1%}", end="")
                
            else:
                # No detection
                cv2.putText(display_image, "Searching for 7x5 checkerboard...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add statistics
            success_rate = (detection_count / total_attempts) * 100 if total_attempts > 0 else 0
            cv2.putText(display_image, f"Success Rate: {success_rate:.1f}% ({detection_count}/{total_attempts})", 
                       (10, display_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show image
            cv2.imshow('RealSense L515 - Checkerboard Detection', display_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current image
                timestamp = int(time.time())
                filename = f"checkerboard_test_{timestamp}.jpg"
                cv2.imwrite(filename, display_image)
                print(f"\\nSaved image: {filename}")
            elif key == ord('c') and ret:
                # Capture detection data
                timestamp = int(time.time())
                print(f"\\n--- Detection Capture {timestamp} ---")
                print(f"Pattern size: {pattern_size}")
                print(f"Square size: {square_size_mm}mm")
                print(f"Corners found: {len(corners_refined)}")
                print(f"Pattern center: ({center[0]:.2f}, {center[1]:.2f})")
                print(f"Pattern coverage: {coverage:.1%}")
                print(f"Corner positions (first 5):")
                for i, corner in enumerate(corners_refined[:5]):
                    print(f"  Corner {i+1}: ({corner[0][0]:.2f}, {corner[0][1]:.2f})")
                if len(corners_refined) > 5:
                    print(f"  ... and {len(corners_refined)-5} more corners")
                print("-" * 40)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        pipeline.stop()
        cv2.destroyAllWindows()
        
        print(f"\\n\\nFinal Statistics:")
        print(f"Total detection attempts: {total_attempts}")
        print(f"Successful detections: {detection_count}")
        if total_attempts > 0:
            print(f"Success rate: {(detection_count/total_attempts)*100:.1f}%")
        print("Test completed.")

if __name__ == "__main__":
    main()