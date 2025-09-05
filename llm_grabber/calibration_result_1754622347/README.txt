CALIBRATION RESULT - 1754622347

IMPORTANT: This calibration result MUST be used with the specific camera intrinsics.

Files:
- eye_in_hand_transform.npy: Main transformation matrix
- camera_intrinsics.txt: Camera intrinsics used during calibration
- calibration_data.json: Complete calibration data including intrinsics

USAGE:
When using this transformation matrix for prediction, you MUST load and use 
the camera intrinsics from camera_intrinsics.txt, NOT from intrinsic.txt
or RealSense SDK intrinsics.

Camera Matrix Used:
[[901.46252441   0.         654.69104004]
 [  0.         901.31671143 355.18649292]
 [  0.           0.           1.        ]]

Distortion Coefficients Used:
[ 1.43461272e-01 -4.80352461e-01  3.87886859e-04 -2.64969975e-04
  4.34053481e-01]

CharUco Board Parameters:
- Size: (7, 5)
- Square Length: 0.04m
- Marker Length: 0.03m

Number of calibration points: 14
