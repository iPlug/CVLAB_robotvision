#!/usr/bin/env python3
"""
Test script to isolate ChArUco detection issues.
"""

import cv2
import numpy as np
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from module.sensors.realsense_manager import RealSenseManager

def test_charuco_detection():
    """Test ChArUco detection with real camera data."""
    print("Testing ChArUco detection...")
    
    # Initialize RealSense camera
    sensor_manager = RealSenseManager(realtime_mode=True, use_imu=False)
    
    if not sensor_manager.initialize():
        print("ERROR: Could not initialize RealSense camera")
        return False
    
    print("Camera initialized successfully")
    
    if not sensor_manager.start():
        print("ERROR: Could not start RealSense camera")
        return False
    
    print("Camera started successfully")
    print(f"Camera is running: {sensor_manager.is_running}")
    
    # Allow camera to stabilize
    import time
    print("Allowing camera to stabilize...")
    time.sleep(2)
    
    # ChArUco parameters for 8x11 pattern
    squares_x = 8  # Number of squares in X direction
    squares_y = 11  # Number of squares in Y direction
    square_length = 25.0  # Size of chessboard squares in mm
    marker_length = 20.0  # Size of ArUco markers in mm (should be smaller than square_length)
    
    # Create ArUco dictionary and ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    charuco_board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    
    print(f"ChArUco board: {squares_x}x{squares_y} squares")
    print(f"Square size: {square_length}mm")
    print(f"Marker size: {marker_length}mm")
    print(f"ArUco dictionary: DICT_6X6_250")
    
    # Statistics
    detection_count = 0
    total_attempts = 0
    
    try:
        print("\nStarting live ChArUco detection...")
        print("Press 'q' to quit, 's' to save image, 'c' to capture detection")
        print("Place ChArUco board in front of camera...")
        
        while True:
            # Get camera frame
            vertices, color_image, metadata = sensor_manager.get_frame()
            
            if color_image is None:
                cv2.waitKey(30)
                continue
            
            total_attempts += 1
            
            # Convert RGB to BGR for OpenCV display
            display_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale for detection
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
            
            # ArUco marker detection
            detector_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
            marker_corners, marker_ids, rejected = detector.detectMarkers(gray_image)
            
            detection_success = False
            
            if marker_ids is not None and len(marker_ids) > 0:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(display_image, marker_corners, marker_ids)
                
                # ChArUco corner detection
                charuco_detector = cv2.aruco.CharucoDetector(charuco_board)
                charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray_image)
                
                if charuco_corners is not None and len(charuco_corners) > 0:
                    detection_count += 1
                    detection_success = True
                    
                    # Draw ChArUco corners
                    cv2.aruco.drawDetectedCornersCharuco(display_image, charuco_corners, charuco_ids)
                    
                    # Calculate metrics
                    total_corners = (squares_x - 1) * (squares_y - 1)
                    coverage = len(charuco_corners) / total_corners
                    
                    # Calculate center
                    center = np.mean(charuco_corners, axis=0)[0]
                    cv2.circle(display_image, tuple(center.astype(int)), 5, (0, 255, 0), -1)
                    
                    # Add detection info
                    cv2.putText(display_image, f"DETECTED: ChArUco {squares_x}x{squares_y}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_image, f"Corners: {len(charuco_corners)}/{total_corners}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_image, f"Coverage: {coverage:.1%}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_image, f"Center: ({center[0]:.1f}, {center[1]:.1f})", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_image, f"Markers: {len(marker_ids)}", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    print(f"\rDetection {detection_count}: Corners={len(charuco_corners)}/{total_corners}, Coverage={coverage:.1%}", end="")
                else:
                    # Markers but no ChArUco corners
                    cv2.putText(display_image, f"Markers detected: {len(marker_ids)}, but no ChArUco corners", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                # No markers detected
                cv2.putText(display_image, f"Searching for ChArUco {squares_x}x{squares_y} pattern...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add statistics
            success_rate = (detection_count / total_attempts) * 100 if total_attempts > 0 else 0
            cv2.putText(display_image, f"Success Rate: {success_rate:.1f}% ({detection_count}/{total_attempts})", 
                       (10, display_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show image
            cv2.imshow('RealSense - ChArUco Detection', display_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current image
                timestamp = int(time.time())
                filename = f"charuco_test_{timestamp}.jpg"
                cv2.imwrite(filename, display_image)
                print(f"\nSaved image: {filename}")
            elif key == ord('c') and detection_success:
                # Capture detection data
                timestamp = int(time.time())
                print(f"\n--- ChArUco Detection Capture {timestamp} ---")
                print(f"Pattern size: {squares_x}x{squares_y}")
                print(f"Square size: {square_length}mm")
                print(f"Marker size: {marker_length}mm")
                print(f"ChArUco corners found: {len(charuco_corners)}")
                print(f"ArUco markers found: {len(marker_ids)}")
                print(f"Pattern coverage: {coverage:.1%}")
                print(f"Pattern center: ({center[0]:.2f}, {center[1]:.2f})")
                if charuco_ids is not None:
                    print(f"Corner IDs: {charuco_ids.flatten()}")
                if marker_ids is not None:
                    print(f"Marker IDs: {marker_ids.flatten()}")
                print("-" * 40)
        
        return detection_count > 0
        
    except Exception as e:
        print(f"ERROR during detection test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return False
    finally:
        sensor_manager.cleanup()
        cv2.destroyAllWindows()
        
        print(f"\n\nFinal Statistics:")
        print(f"Total detection attempts: {total_attempts}")
        print(f"Successful detections: {detection_count}")
        if total_attempts > 0:
            print(f"Success rate: {(detection_count/total_attempts)*100:.1f}%")
        print("ChArUco test completed.")

def test_charuco_generation():
    """Test ChArUco board generation and save to file."""
    print("\n--- ChArUco Board Generation Test ---")
    
    # ChArUco parameters
    squares_x = 8
    squares_y = 11
    square_length = 25.0
    marker_length = 20.0
    
    # Create ArUco dictionary and ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    charuco_board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    
    # Generate board image
    board_size = (squares_x * 100, squares_y * 100)  # 100 pixels per square
    board_image = charuco_board.generateImage(board_size)
    
    # Save board image
    output_file = f"charuco_board_{squares_x}x{squares_y}.png"
    cv2.imwrite(output_file, board_image)
    print(f"ChArUco board saved to: {output_file}")
    
    # Save board parameters
    params_file = f"charuco_board_{squares_x}x{squares_y}_params.json"
    import json
    params = {
        "squares_x": squares_x,
        "squares_y": squares_y,
        "square_length_mm": square_length,
        "marker_length_mm": marker_length,
        "aruco_dict": "DICT_6X6_250",
        "board_size_pixels": board_size,
        "total_squares": squares_x * squares_y,
        "total_corners": (squares_x - 1) * (squares_y - 1),
        "expected_markers": squares_x * squares_y // 2  # Approximate
    }
    
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"ChArUco parameters saved to: {params_file}")
    
    return True

if __name__ == "__main__":
    print("ChArUco Detection Test")
    print("=" * 50)
    
    # Test board generation
    gen_success = test_charuco_generation()
    
    # Test detection
    det_success = test_charuco_detection()
    
    overall_success = gen_success and det_success
    print(f"\nOverall test {'PASSED' if overall_success else 'FAILED'}")
    
    if not overall_success:
        print("\nTroubleshooting tips:")
        print("1. Ensure good lighting without shadows")
        print("2. Hold the ChArUco board steady and flat")
        print("3. Make sure the board is fully visible in camera view")
        print("4. Check that the board parameters match your printed pattern")
        print("5. Verify camera is working and focused properly")
    
    sys.exit(0 if overall_success else 1)