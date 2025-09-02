
(.venv) E:\_S3\LABS\CVLAB_robotvision>E:/_S3/LABS/CVLAB_robotvision/.venv/Scripts/python.exe e:/_S3/LABS/CVLAB_robotvision/charuco_eye_in_hand.py --num-positions 5 --no-filtering
Warning: Pattern position distance (400mm) may be outside robot reach
\nInitializing eye-in-hand CharUco calibration system...
DEBUG: Overriding pattern position from [400, 0, 0] to [400, 0, 0]
Starting eye-in-hand calibration process...
\nIMPORTANT INSTRUCTIONS:
1. Ensure the RealSense camera is securely mounted on the robot
2. Place the CharUco pattern on the table at the specified position
   Pattern position: [400, 0, 0] mm
3. Ensure the pattern will remain stationary throughout calibration
4. Check that lighting is good and even (no shadows on pattern)
5. Ensure the robot workspace is clear for camera movement
\nThe robot will move the camera to multiple positions to view the pattern.
\nStarting calibration automatically...
Initializing CharUco Camera-Robot Calibration System
============================================================
Mode: eye_in_hand
Pattern: 8x11 CharUco (25.0mm squares, 20.0mm markers)
============================================================
2025-08-06 19:11:10 - CharUco calibration process started
2025-08-06 19:11:10 - Mode: eye_in_hand
2025-08-06 19:11:10 - Pattern: 8x11 CharUco (25.0mm squares, 20.0mm markers)
Initializing camera sensor...
Live camera mode
Initializing robot controller...
Transformation matrix loaded: charuco_eye_in_hand_transform.npy
Connecting to myCobot...

1 : COM3 - USB-Enhanced-SERIAL CH9102 (COM3)
Using port: COM3, baud: 115200
myCobot connected successfully!
2025-08-06 19:11:12 - Robot safety parameters: {'min_height': 80, 'max_speed': 35, 'pointing_angles': [0, 180, 0], 'min_update_interval': 0.3}
2025-08-06 19:11:12 - Robot async mode enabled
Initializing CharUco detection...
Quality filtering DISABLED - using permissive detection like test_charuco_detection.py
Loading camera calibration for 3D pose estimation...
Using RealSense factory calibration for 3D pose estimation
Camera matrix fx=601.0, fy=600.9
Principal point: cx=329.8, cy=236.8
2025-08-06 19:11:12 - Using RealSense factory calibration
Camera calibration loaded successfully for CharUco strategy
2025-08-06 19:11:12 - All systems initialized successfully
All systems initialized successfully
Camera calibration ready for 3D pose estimation
\nStep 2: Hand-eye calibration
DEBUG: Loaded 20 pre-recorded positions from e:\_S3\LABS\CVLAB_robotvision\recorded_coords_280.json
DEBUG: Using 5 positions for CharUco calibration
DEBUG: First position: [189.5, -18.7, 294.9, -168.45, 20.91, -142.05]
DEBUG: Last position: [180.3, 0.2, 197.8, -164.02, 25.76, -141.54]
\n============================================================
HAND-EYE CALIBRATION PROCESS
============================================================
Mode: eye_in_hand
Collecting 5 calibration points
Pattern: 8x11 CharUco (25.0mm squares)
============================================================
2025-08-06 19:11:12 - Starting hand-eye calibration with 5 points
\n--- CALIBRATION POINT 1/5 ---
2025-08-06 19:11:12 - Starting calibration point 1/5
Eye-in-hand mode - Robot camera will view fixed CharUco pattern at position 1
\nSTEP: CHARUCO PATTERN SETUP
1. Place the CharUco pattern on the table in a fixed position
2. Ensure it's clearly visible and well-lit
3. Do not move the pattern during the entire calibration process
4. Starting calibration automatically...
Moving robot camera to viewing position: [189.5, -18.7, 294.9, -168.45, 20.91, -142.05]
2025-08-06 19:11:12 - Robot camera position 1: [189.5, -18.7, 294.9, -168.45, 20.91, -142.05]
myCobot movement command sent: ['189.5', '-18.7', '294.9', '-168.4', '20.9', '-142.1'], speed: 30
Robot movement completed
\nCapturing pattern view from position 1...
\nPattern Detection Mode:
- Ensure CharUco pattern is clearly visible
- Press 'c' to capture pattern pose when detected
- Press 'q' to skip this calibration point
- Pattern corners and markers will be outlined when detected
CharUco pattern captured with 3D pose
2025-08-06 19:11:16 - Point 1 captured successfully:
2025-08-06 19:11:16 -   Robot pose: [189.5, -18.7, 294.9, -168.45, 20.91, -142.05]
2025-08-06 19:11:16 -   Pattern pose: {'rvec': array([[-0.21783161],
       [ 0.1090486 ],
       [-0.15467398]]), 'tvec': array([[ -94.77936965],
       [-131.03056959],
       [ 305.48614392]]), 'translation_mm': array([ -94.77936965, -131.03056959,  305.48614392]), 'reprojection_error': 0.0528069806112181, 'quality_score': 0.9947193019388781}
Calibration point 1 captured successfully
\n--- CALIBRATION POINT 2/5 ---
2025-08-06 19:11:16 - Starting calibration point 2/5
Eye-in-hand mode - Robot camera will view fixed CharUco pattern at position 2
Moving robot camera to viewing position: [231.7, -41.7, 243.8, -173.79, 16.86, -136.02]
2025-08-06 19:11:16 - Robot camera position 2: [231.7, -41.7, 243.8, -173.79, 16.86, -136.02]
myCobot movement command sent: ['231.7', '-41.7', '243.8', '-173.8', '16.9', '-136.0'], speed: 30
Robot movement completed
\nCapturing pattern view from position 2...
\nPattern Detection Mode:
- Ensure CharUco pattern is clearly visible
- Press 'c' to capture pattern pose when detected
- Press 'q' to skip this calibration point
- Pattern corners and markers will be outlined when detected
CharUco pattern captured with 3D pose
2025-08-06 19:11:20 - Point 2 captured successfully:
2025-08-06 19:11:20 -   Robot pose: [231.7, -41.7, 243.8, -173.79, 16.86, -136.02]
2025-08-06 19:11:20 -   Pattern pose: {'rvec': array([[-0.08470898],
       [ 0.15630768],
       [-0.03354544]]), 'tvec': array([[ -85.0125973 ],
       [-155.00380665],
       [ 220.96836421]]), 'translation_mm': array([ -85.0125973 , -155.00380665,  220.96836421]), 'reprojection_error': 0.05997824886527577, 'quality_score': 0.9940021751134724}
Calibration point 2 captured successfully
\n--- CALIBRATION POINT 3/5 ---
2025-08-06 19:11:20 - Starting calibration point 3/5
Eye-in-hand mode - Robot camera will view fixed CharUco pattern at position 3
Moving robot camera to viewing position: [243.8, -71.1, 215.8, -168.22, 12.73, -112.52]
2025-08-06 19:11:20 - Robot camera position 3: [243.8, -71.1, 215.8, -168.22, 12.73, -112.52]
myCobot movement command sent: ['243.8', '-71.1', '215.8', '-168.2', '12.7', '-112.5'], speed: 30
Robot movement completed
\nCapturing pattern view from position 3...
\nPattern Detection Mode:
- Ensure CharUco pattern is clearly visible
- Press 'c' to capture pattern pose when detected
- Press 'q' to skip this calibration point
- Pattern corners and markers will be outlined when detected
CharUco pattern captured with 3D pose
2025-08-06 19:11:24 - Point 3 captured successfully:
2025-08-06 19:11:24 -   Robot pose: [243.8, -71.1, 215.8, -168.22, 12.73, -112.52]
2025-08-06 19:11:24 -   Pattern pose: {'rvec': array([[-0.12937948],
       [ 0.13778504],
       [ 0.36284224]]), 'tvec': array([[ -29.55879428],
       [-172.77951974],
       [ 197.99681241]]), 'translation_mm': array([ -29.55879428, -172.77951974,  197.99681241]), 'reprojection_error': 0.10474374211099485, 'quality_score': 0.9895256257889005}
Calibration point 3 captured successfully
\n--- CALIBRATION POINT 4/5 ---
2025-08-06 19:11:24 - Starting calibration point 4/5
Eye-in-hand mode - Robot camera will view fixed CharUco pattern at position 4
Moving robot camera to viewing position: [243.4, -12.6, 236.6, 176.93, 18.46, -161.9]
2025-08-06 19:11:24 - Robot camera position 4: [243.4, -12.6, 236.6, 176.93, 18.46, -161.9]
myCobot movement command sent: ['243.4', '-12.6', '236.6', '176.9', '18.5', '-161.9'], speed: 30
Robot movement completed
\nCapturing pattern view from position 4...
\nPattern Detection Mode:
- Ensure CharUco pattern is clearly visible
- Press 'c' to capture pattern pose when detected
- Press 'q' to skip this calibration point
- Pattern corners and markers will be outlined when detected
CharUco pattern captured with 3D pose
2025-08-06 19:11:29 - Point 4 captured successfully:
2025-08-06 19:11:29 -   Robot pose: [243.4, -12.6, 236.6, 176.93, 18.46, -161.9]
2025-08-06 19:11:29 -   Pattern pose: {'rvec': array([[-0.01750156],
       [ 0.17851558],
       [-0.44476375]]), 'tvec': array([[-136.57526348],
       [-112.9925352 ],
       [ 195.27493892]]), 'translation_mm': array([-136.57526348, -112.9925352 ,  195.27493892]), 'reprojection_error': 0.06671906108470926, 'quality_score': 0.9933280938915291}
Calibration point 4 captured successfully
\n--- CALIBRATION POINT 5/5 ---
2025-08-06 19:11:29 - Starting calibration point 5/5
Eye-in-hand mode - Robot camera will view fixed CharUco pattern at position 5
Moving robot camera to viewing position: [180.3, 0.2, 197.8, -164.02, 25.76, -141.54]
2025-08-06 19:11:29 - Robot camera position 5: [180.3, 0.2, 197.8, -164.02, 25.76, -141.54]
myCobot movement command sent: ['180.3', '0.2', '197.8', '-164.0', '25.8', '-141.5'], speed: 30
Robot movement completed
\nCapturing pattern view from position 5...
\nPattern Detection Mode:
- Ensure CharUco pattern is clearly visible
- Press 'c' to capture pattern pose when detected
- Press 'q' to skip this calibration point
- Pattern corners and markers will be outlined when detected
CharUco pattern captured with 3D pose
2025-08-06 19:11:33 - Point 5 captured successfully:
2025-08-06 19:11:33 -   Robot pose: [180.3, 0.2, 197.8, -164.02, 25.76, -141.54]
2025-08-06 19:11:33 -   Pattern pose: {'rvec': array([[-0.32513444],
       [ 0.07038361],
       [-0.16443559]]), 'tvec': array([[ -98.20543133],
       [-142.95357245],
       [ 240.2278312 ]]), 'translation_mm': array([ -98.20543133, -142.95357245,  240.2278312 ]), 'reprojection_error': 0.0573677689821436, 'quality_score': 0.9942632231017856}
Calibration point 5 captured successfully
\nCalculating hand-eye transformation from 5 point pairs...
2025-08-06 19:11:33 - Calculating hand-eye transformation from 5 pairs
2025-08-06 19:11:33 - Using L515-optimized eye-in-hand calibration workflow
2025-08-06 19:11:33 - Using OpenCV's calibrateHandEye for robot-to-color calibration
2025-08-06 19:11:33 - OpenCV calibrateHandEye completed successfully
2025-08-06 19:11:33 - Rotation matrix determinant: 1.000000 (should be ±1)
Factory extrinsics retrieved successfully:
  Translation: [0.000075, -0.013722, 0.004933] meters
  Rotation determinant: 1.000000 (should be ±1)
2025-08-06 19:11:33 - Successfully retrieved factory-calibrated extrinsics
2025-08-06 19:11:33 - Chaining transformations: T_robot_to_depth = T_robot_to_color * T_color_to_depth
2025-08-06 19:11:33 - L515 factory extrinsics successfully integrated
2025-08-06 19:11:33 - Color-to-depth translation: [ 7.46487349e-05 -1.37223061e-02  4.93308296e-03]
2025-08-06 19:11:33 - Final robot-to-depth transformation calculated
Hand-eye calibration successful
2025-08-06 19:11:33 - Hand-eye calibration successful
2025-08-06 19:11:33 - Transformation matrix saved to: E:\_S3\LABS\CVLAB_robotvision\charuco_eye_in_hand_transform.npy
Transformation matrix saved to: E:\_S3\LABS\CVLAB_robotvision\charuco_eye_in_hand_transform.npy
\n============================================================
VALIDATION: Testing transformation accuracy with training data
============================================================
Validating with 5 training data points...
  Point 1: Error = 79.89mm
  Point 2: Error = 79.89mm
  Point 3: Error = 79.89mm
Validation completed: 5/5 points
Mean error: 79.89mm
Std deviation: 0.00mm
Min/Max error: 79.89mm / 79.89mm
2025-08-06 19:11:33 - Validation results: mean=79.89mm, std=0.00mm, max=79.89mm
\nCHARUCO CALIBRATION COMPLETED SUCCESSFULLY!
Mode: eye_in_hand
Calibration pairs: 5
Camera calibration: Yes
Validation accuracy: 79.89mm ± 0.00mm
Max error: 79.89mm
[GOOD] Calibration accuracy is GOOD (< 100mm)
2025-08-06 19:11:33 - CharUco calibration completed successfully
\nCleaning up CharUco calibration system...
RealSense pipeline stopped
myCobot disconnected
Cleanup complete
\n============================================================
CHARUCO EYE-IN-HAND CALIBRATION COMPLETED SUCCESSFULLY!
============================================================
Transformation matrix saved to: charuco_eye_in_hand_transform.npy
Camera calibration saved to: charuco_camera_calibration.json
Calibration log saved to: charuco_eye_in_hand_calibration.log
\nYou can now use this calibration for:
- Eye-in-hand visual servoing with CharUco patterns
- High-precision robot-guided object inspection
- Dynamic pick and place operations
- Real-time visual feedback control

(.venv) E:\_S3\LABS\CVLAB_robotvision>E:/_S3/LABS/CVLAB_robotvision/.venv/Scripts/python.exe e:/_S3/LABS/CVLAB_robotvision/charuco_eye_in_hand.py --num-positions 5 --no-filtering --pattern-position 200 115 0
\nInitializing eye-in-hand CharUco calibration system...
DEBUG: Overriding pattern position from [400, 0, 0] to [200.0, 115.0, 0.0]
Starting eye-in-hand calibration process...
\nIMPORTANT INSTRUCTIONS:
1. Ensure the RealSense camera is securely mounted on the robot
2. Place the CharUco pattern on the table at the specified position
   Pattern position: [200.0, 115.0, 0.0] mm
3. Ensure the pattern will remain stationary throughout calibration
4. Check that lighting is good and even (no shadows on pattern)
5. Ensure the robot workspace is clear for camera movement
\nThe robot will move the camera to multiple positions to view the pattern.
\nStarting calibration automatically...
Initializing CharUco Camera-Robot Calibration System
============================================================
Mode: eye_in_hand
Pattern: 8x11 CharUco (25.0mm squares, 20.0mm markers)
============================================================
2025-08-06 19:15:20 - CharUco calibration process started
2025-08-06 19:15:20 - Mode: eye_in_hand
2025-08-06 19:15:20 - Pattern: 8x11 CharUco (25.0mm squares, 20.0mm markers)
Initializing camera sensor...
Live camera mode
Initializing robot controller...
Transformation matrix loaded: charuco_eye_in_hand_transform.npy
Connecting to myCobot...

1 : COM3 - USB-Enhanced-SERIAL CH9102 (COM3)
Using port: COM3, baud: 115200
myCobot connected successfully!
2025-08-06 19:15:22 - Robot safety parameters: {'min_height': 80, 'max_speed': 35, 'pointing_angles': [0, 180, 0], 'min_update_interval': 0.3}
2025-08-06 19:15:22 - Robot async mode enabled
Initializing CharUco detection...
Quality filtering DISABLED - using permissive detection like test_charuco_detection.py
Loading camera calibration for 3D pose estimation...
Using RealSense factory calibration for 3D pose estimation
Camera matrix fx=601.0, fy=600.9
Principal point: cx=329.8, cy=236.8
2025-08-06 19:15:22 - Using RealSense factory calibration
Camera calibration loaded successfully for CharUco strategy
2025-08-06 19:15:22 - All systems initialized successfully
All systems initialized successfully
Camera calibration ready for 3D pose estimation
\nStep 2: Hand-eye calibration
DEBUG: Loaded 20 pre-recorded positions from e:\_S3\LABS\CVLAB_robotvision\recorded_coords_280.json
DEBUG: Using 5 positions for CharUco calibration
DEBUG: First position: [189.5, -18.7, 294.9, -168.45, 20.91, -142.05]
DEBUG: Last position: [180.3, 0.2, 197.8, -164.02, 25.76, -141.54]
\n============================================================
HAND-EYE CALIBRATION PROCESS
============================================================
Mode: eye_in_hand
Collecting 5 calibration points
Pattern: 8x11 CharUco (25.0mm squares)
============================================================
2025-08-06 19:15:22 - Starting hand-eye calibration with 5 points
\n--- CALIBRATION POINT 1/5 ---
2025-08-06 19:15:22 - Starting calibration point 1/5
Eye-in-hand mode - Robot camera will view fixed CharUco pattern at position 1
\nSTEP: CHARUCO PATTERN SETUP
1. Place the CharUco pattern on the table in a fixed position
2. Ensure it's clearly visible and well-lit
3. Do not move the pattern during the entire calibration process
4. Starting calibration automatically...
Moving robot camera to viewing position: [189.5, -18.7, 294.9, -168.45, 20.91, -142.05]
2025-08-06 19:15:22 - Robot camera position 1: [189.5, -18.7, 294.9, -168.45, 20.91, -142.05]
myCobot movement command sent: ['189.5', '-18.7', '294.9', '-168.4', '20.9', '-142.1'], speed: 30
Robot movement completed
\nCapturing pattern view from position 1...
\nPattern Detection Mode:
- Ensure CharUco pattern is clearly visible
- Press 'c' to capture pattern pose when detected
- Press 'q' to skip this calibration point
- Pattern corners and markers will be outlined when detected
CharUco pattern captured with 3D pose
2025-08-06 19:15:26 - Point 1 captured successfully:
2025-08-06 19:15:26 -   Robot pose: [189.5, -18.7, 294.9, -168.45, 20.91, -142.05]
2025-08-06 19:15:26 -   Pattern pose: {'rvec': array([[-0.22333057],
       [ 0.07598378],
       [-0.14121336]]), 'tvec': array([[-100.76391031],
       [-129.99262084],
       [ 303.38045489]]), 'translation_mm': array([-100.76391031, -129.99262084,  303.38045489]), 'reprojection_error': 0.059312737114277335, 'quality_score': 0.9940687262885722}
Calibration point 1 captured successfully
\n--- CALIBRATION POINT 2/5 ---
2025-08-06 19:15:26 - Starting calibration point 2/5
Eye-in-hand mode - Robot camera will view fixed CharUco pattern at position 2
Moving robot camera to viewing position: [231.7, -41.7, 243.8, -173.79, 16.86, -136.02]
2025-08-06 19:15:26 - Robot camera position 2: [231.7, -41.7, 243.8, -173.79, 16.86, -136.02]
myCobot movement command sent: ['231.7', '-41.7', '243.8', '-173.8', '16.9', '-136.0'], speed: 30
Robot movement completed
\nCapturing pattern view from position 2...
\nPattern Detection Mode:
- Ensure CharUco pattern is clearly visible
- Press 'c' to capture pattern pose when detected
- Press 'q' to skip this calibration point
- Pattern corners and markers will be outlined when detected
CharUco pattern captured with 3D pose
2025-08-06 19:15:29 - Point 2 captured successfully:
2025-08-06 19:15:29 -   Robot pose: [231.7, -41.7, 243.8, -173.79, 16.86, -136.02]
2025-08-06 19:15:29 -   Pattern pose: {'rvec': array([[-0.08494979],
       [ 0.13888831],
       [-0.03140301]]), 'tvec': array([[ -88.93742033],
       [-154.85769093],
       [ 219.27492189]]), 'translation_mm': array([ -88.93742033, -154.85769093,  219.27492189]), 'reprojection_error': 0.07293290612208439, 'quality_score': 0.9927067093877916}
Calibration point 2 captured successfully
\n--- CALIBRATION POINT 3/5 ---
2025-08-06 19:15:29 - Starting calibration point 3/5
Eye-in-hand mode - Robot camera will view fixed CharUco pattern at position 3
Moving robot camera to viewing position: [243.8, -71.1, 215.8, -168.22, 12.73, -112.52]
2025-08-06 19:15:29 - Robot camera position 3: [243.8, -71.1, 215.8, -168.22, 12.73, -112.52]
myCobot movement command sent: ['243.8', '-71.1', '215.8', '-168.2', '12.7', '-112.5'], speed: 30
Robot movement completed
\nCapturing pattern view from position 3...
\nPattern Detection Mode:
- Ensure CharUco pattern is clearly visible
- Press 'c' to capture pattern pose when detected
- Press 'q' to skip this calibration point
- Pattern corners and markers will be outlined when detected
CharUco pattern captured with 3D pose
2025-08-06 19:15:34 - Point 3 captured successfully:
2025-08-06 19:15:34 -   Robot pose: [243.8, -71.1, 215.8, -168.22, 12.73, -112.52]
2025-08-06 19:15:34 -   Pattern pose: {'rvec': array([[-0.12615013],
       [ 0.13827771],
       [ 0.36156414]]), 'tvec': array([[ -29.7636695 ],
       [-173.4643104 ],
       [ 197.28416386]]), 'translation_mm': array([ -29.7636695 , -173.4643104 ,  197.28416386]), 'reprojection_error': 0.08837315196857413, 'quality_score': 0.9911626848031426}
Calibration point 3 captured successfully
\n--- CALIBRATION POINT 4/5 ---
2025-08-06 19:15:34 - Starting calibration point 4/5
Eye-in-hand mode - Robot camera will view fixed CharUco pattern at position 4
Moving robot camera to viewing position: [243.4, -12.6, 236.6, 176.93, 18.46, -161.9]
2025-08-06 19:15:34 - Robot camera position 4: [243.4, -12.6, 236.6, 176.93, 18.46, -161.9]
myCobot movement command sent: ['243.4', '-12.6', '236.6', '176.9', '18.5', '-161.9'], speed: 30
Robot movement completed
\nCapturing pattern view from position 4...
\nPattern Detection Mode:
- Ensure CharUco pattern is clearly visible
- Press 'c' to capture pattern pose when detected
- Press 'q' to skip this calibration point
- Pattern corners and markers will be outlined when detected
CharUco pattern captured with 3D pose
2025-08-06 19:15:38 - Point 4 captured successfully:
2025-08-06 19:15:38 -   Robot pose: [243.4, -12.6, 236.6, 176.93, 18.46, -161.9]
2025-08-06 19:15:38 -   Pattern pose: {'rvec': array([[-0.01632416],
       [ 0.18359022],
       [-0.43561977]]), 'tvec': array([[-132.57149103],
       [-114.27576992],
       [ 196.33466098]]), 'translation_mm': array([-132.57149103, -114.27576992,  196.33466098]), 'reprojection_error': 0.08382911735346923, 'quality_score': 0.9916170882646531}
Calibration point 4 captured successfully
\n--- CALIBRATION POINT 5/5 ---
2025-08-06 19:15:38 - Starting calibration point 5/5
Eye-in-hand mode - Robot camera will view fixed CharUco pattern at position 5
Moving robot camera to viewing position: [180.3, 0.2, 197.8, -164.02, 25.76, -141.54]
2025-08-06 19:15:38 - Robot camera position 5: [180.3, 0.2, 197.8, -164.02, 25.76, -141.54]
myCobot movement command sent: ['180.3', '0.2', '197.8', '-164.0', '25.8', '-141.5'], speed: 30
Robot movement completed
\nCapturing pattern view from position 5...
\nPattern Detection Mode:
- Ensure CharUco pattern is clearly visible
- Press 'c' to capture pattern pose when detected
- Press 'q' to skip this calibration point
- Pattern corners and markers will be outlined when detected
CharUco pattern captured with 3D pose
2025-08-06 19:15:42 - Point 5 captured successfully:
2025-08-06 19:15:42 -   Robot pose: [180.3, 0.2, 197.8, -164.02, 25.76, -141.54]
2025-08-06 19:15:42 -   Pattern pose: {'rvec': array([[-0.31516909],
       [ 0.07040536],
       [-0.16025167]]), 'tvec': array([[ -96.75953887],
       [-144.64029298],
       [ 238.3497879 ]]), 'translation_mm': array([ -96.75953887, -144.64029298,  238.3497879 ]), 'reprojection_error': 0.09226230294991428, 'quality_score': 0.9907737697050085}
Calibration point 5 captured successfully
\nCalculating hand-eye transformation from 5 point pairs...
2025-08-06 19:15:42 - Calculating hand-eye transformation from 5 pairs
2025-08-06 19:15:42 - Using L515-optimized eye-in-hand calibration workflow
2025-08-06 19:15:42 - Using OpenCV's calibrateHandEye for robot-to-color calibration
2025-08-06 19:15:42 - OpenCV calibrateHandEye completed successfully
2025-08-06 19:15:42 - Rotation matrix determinant: 1.000000 (should be ±1)
Factory extrinsics retrieved successfully:
  Translation: [0.000075, -0.013722, 0.004933] meters
  Rotation determinant: 1.000000 (should be ±1)
2025-08-06 19:15:42 - Successfully retrieved factory-calibrated extrinsics
2025-08-06 19:15:42 - Chaining transformations: T_robot_to_depth = T_robot_to_color * T_color_to_depth
2025-08-06 19:15:42 - L515 factory extrinsics successfully integrated
2025-08-06 19:15:42 - Color-to-depth translation: [ 7.46487349e-05 -1.37223061e-02  4.93308296e-03]
2025-08-06 19:15:42 - Final robot-to-depth transformation calculated
Hand-eye calibration successful
2025-08-06 19:15:42 - Hand-eye calibration successful
2025-08-06 19:15:42 - Transformation matrix saved to: E:\_S3\LABS\CVLAB_robotvision\charuco_eye_in_hand_transform.npy
Transformation matrix saved to: E:\_S3\LABS\CVLAB_robotvision\charuco_eye_in_hand_transform.npy
\n============================================================
VALIDATION: Testing transformation accuracy with training data
============================================================
Validating with 5 training data points...
  Point 1: Error = 78.08mm
  Point 2: Error = 78.08mm
  Point 3: Error = 78.08mm
Validation completed: 5/5 points
Mean error: 78.08mm
Std deviation: 0.00mm
Min/Max error: 78.08mm / 78.08mm
2025-08-06 19:15:42 - Validation results: mean=78.08mm, std=0.00mm, max=78.08mm
\nCHARUCO CALIBRATION COMPLETED SUCCESSFULLY!
Mode: eye_in_hand
Calibration pairs: 5
Camera calibration: Yes
Validation accuracy: 78.08mm ± 0.00mm
Max error: 78.08mm
[GOOD] Calibration accuracy is GOOD (< 100mm)
2025-08-06 19:15:42 - CharUco calibration completed successfully
\nCleaning up CharUco calibration system...
RealSense pipeline stopped
myCobot disconnected
Cleanup complete
\n============================================================
CHARUCO EYE-IN-HAND CALIBRATION COMPLETED SUCCESSFULLY!
============================================================
Transformation matrix saved to: charuco_eye_in_hand_transform.npy
Camera calibration saved to: charuco_camera_calibration.json
Calibration log saved to: charuco_eye_in_hand_calibration.log
\nYou can now use this calibration for:
- Eye-in-hand visual servoing with CharUco patterns
- High-precision robot-guided object inspection
- Dynamic pick and place operations
- Real-time visual feedback control