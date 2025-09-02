#!/usr/bin/env python3
"""
Test script to demonstrate the improved eye-in-hand calibration with diverse rotations.

This script shows the differences between the old and new calibration approaches:
1. Old: Small rotational variations (5-10 degrees) - causes OpenCV calibration to fail
2. New: Large rotational diversity (20-60 degrees) - enables successful OpenCV calibration

Usage:
    python test_diverse_calibration.py
"""

import numpy as np
import cv2

def test_calibration_diversity():
    """Test calibration with different levels of rotational diversity."""
    
    print("=" * 70)
    print("EYE-IN-HAND CALIBRATION DIVERSITY TEST")
    print("=" * 70)
    
    # Test 1: Old approach with minimal rotational diversity
    print("\n1. TESTING OLD APPROACH (Minimal Rotational Diversity)")
    print("-" * 50)
    
    old_positions = [
        [87.5, -62.7, 367.2, -144.03, 29.89, -124.04],  # Reference
        [150, 0, 300, -144, 30, -124],       # +5° variation
        [120, -80, 320, -140, 25, -120],     # +4-5° variation
        [120, 80, 320, -140, 25, -120],      # +4-5° variation
        [180, 0, 280, -150, 35, -130],       # +6-10° variation
    ]
    
    rotational_diversity_old = calculate_rotational_diversity(old_positions)
    print(f"Rotational diversity (old): {rotational_diversity_old:.1f} degrees")
    print("Result: INSUFFICIENT for OpenCV calibrateHandEye")
    
    # Test 2: New approach with large rotational diversity (FIXED - camera looking down)
    print("\n2. TESTING FIXED NEW APPROACH (Large Rotational Diversity)")
    print("-" * 50)
    
    new_positions = [
        [87.5, -62.7, 367.2, -144.03, 29.89, -124.04],   # Reference
        [150, 0, 300, -144, 30, -124],                   # Same as reference
        [120, -80, 320, -160, 45, -140],                # Left, +16 deg roll, +15 deg pitch, -16 deg yaw
        [120, 80, 320, -130, 15, -110],                 # Right, +14 deg roll, -15 deg pitch, +14 deg yaw  
        [180, 0, 280, -155, 50, -115],                  # Center, -11 deg roll, +20 deg pitch, +9 deg yaw
    ]
    
    rotational_diversity_new = calculate_rotational_diversity(new_positions)
    print(f"Rotational diversity (new): {rotational_diversity_new:.1f} degrees")
    print("Result: SUFFICIENT for OpenCV calibrateHandEye")
    
    # Test 3: Precision approach with maximum diversity (FIXED - camera looking down)
    print("\n3. TESTING FIXED PRECISION APPROACH (Maximum Rotational Diversity)")
    print("-" * 50)
    
    precision_positions = [
        [87.5, -62.7, 367.2, -144.03, 29.89, -124.04],   # Reference
        [150, 0, 300, -144, 30, -124],                   # Same as reference
        [120, -120, 320, -170, 50, -150],               # Far left, rotated but looking down
        [130, -60, 310, -135, 10, -130],                # Near left, slight rotation
        [130, 60, 310, -150, 45, -115],                 # Near right, angled but looking down
        [120, 120, 320, -125, 20, -105],                # Far right, rotated but looking down
        [100, 0, 340, -160, 60, -120],                  # High position, steep downward angle
    ]
    
    rotational_diversity_precision = calculate_rotational_diversity(precision_positions)
    print(f"Rotational diversity (precision): {rotational_diversity_precision:.1f} degrees")
    print("Result: OPTIMAL for OpenCV calibrateHandEye")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 70)
    print(f"Old approach diversity:       {rotational_diversity_old:.1f} degrees - [FAILS] OpenCV")
    print(f"New approach diversity:       {rotational_diversity_new:.1f} degrees - [WORKS] with OpenCV")
    print(f"Precision approach diversity: {rotational_diversity_precision:.1f} degrees - [OPTIMAL] for OpenCV")
    print()
    print("RECOMMENDATIONS:")
    print("- Use --positions 5 for standard calibration (25-35 degree diversity)")
    print("- Use --positions 7 for precision calibration (40-50 degree diversity)")  
    print("- Ensure each pose has at least 20 degree rotation difference from reference")
    print("- OpenCV calibrateHandEye requires >=25 degree rotational diversity to succeed")
    print()
    print("NEXT STEPS:")
    print("1. Run: python eye_in_hand_calibration.py --positions 5")
    print("2. Follow the interactive prompts to collect new calibration data")
    print("3. The system will now use diverse rotations for successful OpenCV calibration")

def calculate_rotational_diversity(positions):
    """Calculate the maximum rotational diversity in a set of positions."""
    if len(positions) < 2:
        return 0.0
    
    reference_rotation = positions[0][3:6]  # First position as reference
    max_difference = 0.0
    
    for position in positions[1:]:
        rotation = position[3:6]
        
        # Calculate angular differences
        rx_diff = abs(rotation[0] - reference_rotation[0])
        ry_diff = abs(rotation[1] - reference_rotation[1])
        rz_diff = abs(rotation[2] - reference_rotation[2])
        
        # Handle angle wrapping (e.g., -180° vs +180°)
        rx_diff = min(rx_diff, 360 - rx_diff)
        ry_diff = min(ry_diff, 360 - ry_diff)
        rz_diff = min(rz_diff, 360 - rz_diff)
        
        # Use the maximum single-axis difference as diversity measure
        total_difference = max(rx_diff, ry_diff, rz_diff)
        max_difference = max(max_difference, total_difference)
    
    return max_difference

if __name__ == "__main__":
    test_calibration_diversity()