#!/usr/bin/env python3
"""
Analysis of current camera intrinsics vs your actual L515 camera.
This explains why you need to use YOUR camera's intrinsics, not GitHub ones.
"""

import numpy as np

print("=== CAMERA INTRINSICS ANALYSIS ===")
print()

# Your actual RealSense L515 intrinsics (measured from your camera)
your_l515_intrinsics = np.array([
    [901.4625244140625, 0.0, 654.6910400390625],
    [0.0, 901.3167114257812, 355.1864929199219],
    [0.0, 0.0, 1.0]
])

# GitHub repository intrinsics (from unknown camera)
github_intrinsics = np.array([
    [1.363386230468750000e+03, 0.000000000000000000e+00, 9.487056884765625000e+02],
    [0.000000000000000000e+00, 1.361707641601562500e+03, 5.192871704101562500e+02],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
])

print("YOUR ACTUAL L515 CAMERA INTRINSICS:")
print(f"  fx={your_l515_intrinsics[0,0]:.2f}, fy={your_l515_intrinsics[1,1]:.2f}")
print(f"  cx={your_l515_intrinsics[0,2]:.2f}, cy={your_l515_intrinsics[1,2]:.2f}")
print(f"  Serial: f1420847")
print(f"  Resolution: 1280x720")
print(f"  Source: YOUR actual camera (RealSense SDK)")
print()

print("GITHUB REPOSITORY INTRINSICS (intrinsic.txt):")
print(f"  fx={github_intrinsics[0,0]:.2f}, fy={github_intrinsics[1,1]:.2f}")
print(f"  cx={github_intrinsics[0,2]:.2f}, cy={github_intrinsics[1,2]:.2f}")
print(f"  Serial: UNKNOWN")
print(f"  Resolution: UNKNOWN (possibly different)")
print(f"  Source: GitHub repository (unknown camera)")
print()

# Calculate differences
fx_diff = github_intrinsics[0,0] - your_l515_intrinsics[0,0]
fy_diff = github_intrinsics[1,1] - your_l515_intrinsics[1,1]
cx_diff = github_intrinsics[0,2] - your_l515_intrinsics[0,2]
cy_diff = github_intrinsics[1,2] - your_l515_intrinsics[1,2]

print("DIFFERENCES (GitHub - Your L515):")
print(f"  Δfx = {fx_diff:+.1f} ({fx_diff/your_l515_intrinsics[0,0]*100:+.1f}%)")
print(f"  Δfy = {fy_diff:+.1f} ({fy_diff/your_l515_intrinsics[1,1]*100:+.1f}%)")
print(f"  Δcx = {cx_diff:+.1f} ({cx_diff/your_l515_intrinsics[0,2]*100:+.1f}%)")
print(f"  Δcy = {cy_diff:+.1f} ({cy_diff/your_l515_intrinsics[1,2]*100:+.1f}%)")
print()

print("ANALYSIS:")
if abs(fx_diff/your_l515_intrinsics[0,0]) > 0.3:  # >30% difference
    print("🚨 MAJOR DIFFERENCES DETECTED!")
    print("   GitHub intrinsics are from a VERY different camera/setup")
    if github_intrinsics[0,0] > your_l515_intrinsics[0,0]:
        print("   GitHub camera: Higher focal length = closer working distance or different lens")
    else:
        print("   GitHub camera: Lower focal length = further working distance or different lens")
    print()
    print("🎯 LIKELY EXPLANATIONS:")
    print("   • Different camera model (not L515)")
    print("   • Different resolution (focal length scales with resolution)")
    print("   • Different lens calibration")
    print("   • Different working distance calibration")
    print()
    print("❌ USING GITHUB INTRINSICS = WRONG CAMERA MODEL!")
else:
    print("✅ Reasonable differences - might be same camera type")

print()
print("🏆 RECOMMENDATION:")
print("   Use YOUR L515 intrinsics for accurate results:")
print("   1. Better accuracy for YOUR specific camera")
print("   2. Correct resolution scaling")
print("   3. Factory-calibrated for your hardware")
print()
print("📝 NEXT STEPS:")
print("   1. Test both intrinsics side by side")
print("   2. Measure actual CharUco position vs predicted")
print("   3. Compare accuracy: GitHub intrinsics vs Your L515 intrinsics")
print("   4. Use whichever gives better real-world accuracy")

# Guess the GitHub camera specs
print()
print("🔍 GUESSING GITHUB CAMERA SPECS:")
fx_ratio = github_intrinsics[0,0] / your_l515_intrinsics[0,0]
print(f"   Focal length ratio: {fx_ratio:.2f}x")

if fx_ratio > 1.4:
    print("   Likely: Higher resolution than 1280x720")
    estimated_width = int(1280 * fx_ratio)
    estimated_height = int(720 * fx_ratio)
    print(f"   Estimated resolution: ~{estimated_width}x{estimated_height}")
    print("   OR: Different camera model with different optics")
elif fx_ratio < 0.8:
    print("   Likely: Lower resolution than 1280x720")
    estimated_width = int(1280 * fx_ratio)
    estimated_height = int(720 * fx_ratio)
    print(f"   Estimated resolution: ~{estimated_width}x{estimated_height}")
else:
    print("   Likely: Similar resolution, different camera model")