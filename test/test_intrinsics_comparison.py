#!/usr/bin/env python3
"""
Test script to compare RealSense SDK intrinsics vs Calibration intrinsics
for CharUco position prediction accuracy.
"""

import numpy as np
import cv2

# Load the same transformation matrix used by both systems
eye_in_hand_transform = np.load('charuco-pose-estimation/calibration_result_1754539897/eye_in_hand_transform.npy')

print("=== INTRINSICS COMPARISON TEST ===")
print("Testing CharUco position prediction accuracy with different camera intrinsics")
print()

# RealSense SDK intrinsics (what we just switched to)
realsense_intrinsics = np.array([
    [901.4625244140625, 0.0, 654.6910400390625],
    [0.0, 901.3167114257812, 355.1864929199219],
    [0.0, 0.0, 1.0]
])

# Calibration intrinsics (what we used before that achieved 2-30mm accuracy)
calibration_intrinsics = np.loadtxt('charuco-pose-estimation/intrinsic.txt')

print("REALSENSE SDK INTRINSICS:")
print(f"  fx={realsense_intrinsics[0,0]:.2f}, fy={realsense_intrinsics[1,1]:.2f}")
print(f"  cx={realsense_intrinsics[0,2]:.2f}, cy={realsense_intrinsics[1,2]:.2f}")
print(f"Matrix:\n{realsense_intrinsics}")
print()

print("CALIBRATION INTRINSICS:")
print(f"  fx={calibration_intrinsics[0,0]:.2f}, fy={calibration_intrinsics[1,1]:.2f}")
print(f"  cx={calibration_intrinsics[0,2]:.2f}, cy={calibration_intrinsics[1,2]:.2f}")
print(f"Matrix:\n{calibration_intrinsics}")
print()

print("KEY DIFFERENCES:")
fx_diff = realsense_intrinsics[0,0] - calibration_intrinsics[0,0]
fy_diff = realsense_intrinsics[1,1] - calibration_intrinsics[1,1]
cx_diff = realsense_intrinsics[0,2] - calibration_intrinsics[0,2]
cy_diff = realsense_intrinsics[1,2] - calibration_intrinsics[1,2]

print(f"  Δfx = {fx_diff:+.1f} ({fx_diff/calibration_intrinsics[0,0]*100:+.1f}%)")
print(f"  Δfy = {fy_diff:+.1f} ({fy_diff/calibration_intrinsics[1,1]*100:+.1f}%)")  
print(f"  Δcx = {cx_diff:+.1f} ({cx_diff/calibration_intrinsics[0,2]*100:+.1f}%)")
print(f"  Δcy = {cy_diff:+.1f} ({cy_diff/calibration_intrinsics[1,2]*100:+.1f}%)")
print()

print("EXPECTED IMPACT:")
print("• Focal length differences (fx,fy) affect depth/distance calculations")  
print("• Principal point differences (cx,cy) affect lateral position accuracy")
print("• RealSense SDK uses factory calibration at single distance")
print("• Custom calibration optimized for CharUco detection at working distance")
print()

print("PREDICTION:")
if abs(fx_diff) > 200 or abs(fy_diff) > 200:
    print("❌ LARGE focal length differences detected!")
    print("   Expected: Lower accuracy with RealSense SDK intrinsics")
    print("   Distance errors likely 100mm+ vs 2-30mm with calibration intrinsics")
else:
    print("✅ Moderate focal length differences")
    print("   Expected: Some accuracy loss but may still work reasonably")

print()
print("RECOMMENDATION:")
print("• Test both systems side by side with CharUco pattern at known position")
print("• Compare predicted positions vs actual measured positions")  
print("• Calibration intrinsics should provide better accuracy for robotics applications")