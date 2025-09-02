#!/usr/bin/env python3
"""
Test ChArUco detection on static images (without camera).
"""

import cv2
import numpy as np
import sys
import os

def test_charuco_on_generated_board():
    """Test ChArUco detection on the generated board image."""
    print("Testing ChArUco detection on generated board...")
    
    # ChArUco parameters matching the generated board
    squares_x = 8
    squares_y = 11
    square_length = 25.0
    marker_length = 20.0
    
    # Create ArUco dictionary and ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    charuco_board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    
    # Load the generated board image
    board_image_file = "charuco_board_8x11.png"
    
    if not os.path.exists(board_image_file):
        print(f"ERROR: {board_image_file} not found. Run test_charuco_detection.py first.")
        return False
    
    # Load and prepare image
    board_image = cv2.imread(board_image_file, cv2.IMREAD_GRAYSCALE)
    if board_image is None:
        print(f"ERROR: Could not load {board_image_file}")
        return False
    
    print(f"Loaded board image: {board_image.shape}")
    
    # Test ArUco marker detection
    print("\n--- ArUco Marker Detection ---")
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    marker_corners, marker_ids, rejected = detector.detectMarkers(board_image)
    
    if marker_ids is not None and len(marker_ids) > 0:
        print(f"Found {len(marker_ids)} ArUco markers")
        print(f"Marker IDs: {marker_ids.flatten()}")
    else:
        print("No ArUco markers detected")
        return False
    
    # Test ChArUco corner detection
    print("\n--- ChArUco Corner Detection ---")
    charuco_detector = cv2.aruco.CharucoDetector(charuco_board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(board_image)
    ret = len(charuco_corners) if charuco_corners is not None else 0
    
    if ret > 0:
        print(f"ChArUco detection successful!")
        print(f"Found {ret} ChArUco corners")
        
        # Calculate quality metrics
        total_corners = (squares_x - 1) * (squares_y - 1)
        coverage = ret / total_corners
        print(f"Corner coverage: {coverage:.2%} ({ret}/{total_corners})")
        
        # Create visualization
        vis_image = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
        
        # Draw detected markers
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(vis_image, marker_corners, marker_ids)
        
        # Draw detected ChArUco corners
        if charuco_corners is not None:
            cv2.aruco.drawDetectedCornersCharuco(vis_image, charuco_corners, charuco_ids, (0, 255, 0))
        
        # Save visualization
        output_file = "charuco_detection_result.png"
        cv2.imwrite(output_file, vis_image)
        print(f"Detection visualization saved to: {output_file}")
        
        return True
    else:
        print("No ChArUco corners detected")
        return False

def test_charuco_with_distortion():
    """Test ChArUco detection with simulated camera distortion."""
    print("\n--- Testing with Simulated Camera Distortion ---")
    
    # Load clean board
    board_image = cv2.imread("charuco_board_8x11.png", cv2.IMREAD_GRAYSCALE)
    if board_image is None:
        print("Cannot load board image")
        return False
    
    # Add some realistic camera effects
    # 1. Add slight blur (camera focus)
    blurred = cv2.GaussianBlur(board_image, (3, 3), 0.5)
    
    # 2. Add noise
    noise = np.random.normal(0, 10, blurred.shape).astype(np.int16)
    noisy = cv2.add(blurred.astype(np.int16), noise)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # 3. Adjust brightness/contrast slightly
    adjusted = cv2.convertScaleAbs(noisy, alpha=0.9, beta=10)
    
    # Test detection on modified image
    squares_x, squares_y = 8, 11
    square_length, marker_length = 25.0, 20.0
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    charuco_board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    charuco_detector = cv2.aruco.CharucoDetector(charuco_board)
    
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(adjusted)
    ret = len(charuco_corners) if charuco_corners is not None else 0
    
    print(f"Detection with camera effects: {ret} corners found")
    
    # Save the modified test image
    cv2.imwrite("charuco_test_distorted.png", adjusted)
    print("Distorted test image saved to: charuco_test_distorted.png")
    
    return ret > 0

if __name__ == "__main__":
    print("ChArUco Static Detection Test")
    print("=" * 50)
    
    # Test on clean generated board
    success1 = test_charuco_on_generated_board()
    
    # Test with simulated camera effects
    success2 = test_charuco_with_distortion()
    
    overall_success = success1 and success2
    print(f"\nOverall static test {'PASSED' if overall_success else 'FAILED'}")
    
    if success1:
        print("\nGenerated files:")
        print("- charuco_board_8x11.png (ChArUco pattern)")
        print("- charuco_board_8x11_params.json (Pattern parameters)")
        print("- charuco_detection_result.png (Detection visualization)")
        print("- charuco_test_distorted.png (Test with camera effects)")
        
        print("\nNext steps:")
        print("1. Print charuco_board_8x11.png on paper")
        print("2. Use test_charuco_detection.py with real camera")
        print("3. Ensure good lighting and steady positioning")
    
    sys.exit(0 if overall_success else 1)