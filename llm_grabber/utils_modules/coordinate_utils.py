#!/usr/bin/env python3
"""
Coordinate utilities for CharUco pose estimation applications.
Handles coordinate transformations and pose handling.
"""

import numpy as np
import cv2
from cv2 import aruco
from scipy.spatial.transform import Rotation


class CoordinateUtils:
    """Utilities for coordinate transformations and pose handling."""
    
    def __init__(self, board_size=(7, 5), square_length=0.039, marker_length=0.029):
        """Initialize coordinate utilities."""
        self.board_size = board_size
        self.square_length = square_length
        self.marker_length = marker_length
        self.recent_positions = []
        self.position_history_size = 10
        
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.board = aruco.CharucoBoard(board_size, square_length, marker_length, dictionary)
    
    def pose_to_transform_matrix(self, pose):
        """Convert [x, y, z, rx, ry, rz] pose to 4x4 transformation matrix."""
        x, y, z, rx, ry, rz = pose
        
        x_m, y_m, z_m = x/1000.0, y/1000.0, z/1000.0
        rx_rad, ry_rad, rz_rad = np.radians([rx, ry, rz])
        
        rotation = Rotation.from_euler('xyz', [rx_rad, ry_rad, rz_rad])
        rot_matrix = rotation.as_matrix()
        
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = [x_m, y_m, z_m]
        
        return transform
    
    def charuco_pose_to_transform_matrix(self, rvec, tvec):
        """Convert CharUco rvec, tvec to 4x4 transformation matrix with coordinate correction."""
        rot_matrix, _ = cv2.Rodrigues(rvec)
        
        correction_matrix = np.array([
            [0, -1, 0],
            [1,  0, 0], 
            [0,  0, 1]
        ])
        
        corrected_rot_matrix = rot_matrix @ correction_matrix
        
        transform = np.eye(4)
        transform[:3, :3] = corrected_rot_matrix
        transform[:3, 3] = tvec.flatten()
        
        return transform
    
    def update_position_history(self, position):
        """Update position history for stability analysis."""
        self.recent_positions.append(position.copy())
        if len(self.recent_positions) > self.position_history_size:
            self.recent_positions.pop(0)
    
    def get_position_stability(self):
        """Get position stability metrics."""
        if len(self.recent_positions) < 2:
            return None, None
        
        positions = np.array(self.recent_positions)
        mean_pos = np.mean(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        return mean_pos, std_pos
    
    def match_image_points_corrected(self, charuco_corners, charuco_ids):
        """
        Match image points with corrected coordinate system (bottom-left origin, outward Z).
        This matches the notebook's coordinate system and standard robotics convention.
        """
        objPoints = []
        imgPoints = []
        
        for i in range(len(charuco_ids)):
            index = charuco_ids[i]
            corner_pos = self.board.getChessboardCorners()[index]
            
            corrected_pos = corner_pos.copy()
            corrected_pos[0][1] = self.board.getRightBottomCorner()[1] - corrected_pos[0][1]
            
            objPoints.append(corrected_pos)
            imgPoints.append(charuco_corners[i])
        
        return np.array(objPoints), np.array(imgPoints)
    
    def convert_to_center_origin_simple(self, tvec, rvec):
        """
        Convert bottom-left origin tvec to center origin using simple offset approach.
        
        Args:
            tvec: Translation vector from bottom-left origin detection
            rvec: Rotation vector from detection
            
        Returns:
            tvec_center: Translation vector for center origin
        """
        center_x = (self.board_size[0] - 1) * self.square_length / 2 + self.square_length / 2
        center_y = (self.board_size[1] - 1) * self.square_length / 2 + self.square_length / 2
        offset_m = np.array([center_x, center_y, 0.0])
        
        rot_matrix, _ = cv2.Rodrigues(rvec)
        offset_in_camera = rot_matrix @ offset_m
        
        tvec_center = tvec.flatten() + offset_in_camera
        
        return tvec_center.reshape(-1, 1)