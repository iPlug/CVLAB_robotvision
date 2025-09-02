#!/usr/bin/env python3
"""
Camera Preview Utilities

Robust camera preview functionality for robot positioning scripts.
Uses modular RealSense manager with proper timeout handling.
"""

import cv2
import numpy as np
import time
import threading
import sys
import os

# Add module path for RealSense manager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import modular manager, fallback to direct RealSense
try:
    from module.sensors.realsense_manager import RealSenseManager
    MODULAR_AVAILABLE = True
except ImportError:
    MODULAR_AVAILABLE = False
    
# Direct RealSense for fallback
import pyrealsense2 as rs


class CameraPreview:
    """Managed camera preview with robust frame handling."""
    
    def __init__(self):
        self.sensor_manager = None
        self.pipeline = None
        self.config = None
        self.use_modular = False
        self.running = False
        self.thread = None
        self.sequence_start_time = None
        self.pre_delay = 3
        self.post_delay = 3
        self.robot_started = False
        self.robot_completed = False
        
    def start(self):
        """Start camera preview in background thread."""
        if self.running:
            return
            
        # Use direct RealSense approach (proven to work)
        # Skip modular approach for now due to initialization issues
        self.use_modular = False
        
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.pipeline.start(self.config)
            print("Camera preview started (direct RealSense)")
        except Exception as e:
            print(f"Direct RealSense error: {e}")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.thread.start()
        
        return True
            
    def stop(self):
        """Stop camera preview and cleanup."""
        if not self.running:
            return
            
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
            
        if self.pipeline:
            self.pipeline.stop()
            
        cv2.destroyAllWindows()
        print("Camera preview stopped")
        
    def set_sequence_timing(self, pre_delay, post_delay):
        """Set timing parameters for robot sequence."""
        self.pre_delay = pre_delay
        self.post_delay = post_delay
        self.sequence_start_time = time.time()
        
    def mark_robot_started(self):
        """Mark that robot movement has started."""
        self.robot_started = True
        
    def mark_robot_completed(self):
        """Mark that robot movement has completed."""
        self.robot_completed = True
        
    def _get_sequence_status(self):
        """Get current status message and timing info."""
        if not self.sequence_start_time:
            return "Camera preview active", (0, 255, 0)
            
        elapsed = time.time() - self.sequence_start_time
        
        if not self.robot_started:
            # Pre-robot phase
            remaining = self.pre_delay - elapsed
            if remaining > 0:
                return f"Robot starts in {remaining:.1f}s", (255, 255, 0)
            else:
                return "Starting robot movement...", (0, 255, 255)
                
        elif not self.robot_completed:
            # Robot moving phase
            return "Robot moving...", (0, 165, 255)
            
        else:
            # Post-robot phase
            robot_complete_time = elapsed  # Approximate
            post_elapsed = time.time() - (self.sequence_start_time + robot_complete_time)
            remaining = self.post_delay - post_elapsed
            if remaining > 0:
                return f"Camera closing in {remaining:.1f}s", (255, 255, 0)
            else:
                return "Sequence complete", (0, 255, 0)
        
    def _preview_loop(self):
        """Internal preview loop with robust frame handling."""
        consecutive_errors = 0
        max_consecutive_errors = 50  # Allow more tolerance for frame drops
        
        while self.running:
            color_image = None
            
            try:
                # Direct RealSense approach with timeout handling
                frames_result = self.pipeline.try_wait_for_frames(timeout_ms=100)
                if frames_result[0]:  # frames_available
                    frames = frames_result[1]
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_image = np.asanyarray(color_frame.get_data())
                    else:
                        color_image = None
                else:
                    color_image = None
                
                if color_image is not None:
                    # Reset error counter on successful frame
                    consecutive_errors = 0
                    
                    # Add status overlay
                    display_image = color_image.copy()
                    status_msg, color = self._get_sequence_status()
                    
                    cv2.putText(display_image, status_msg, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Add timestamp
                    timestamp = time.strftime("%H:%M:%S")
                    cv2.putText(display_image, f"Time: {timestamp}", (10, display_image.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('Robot Camera Preview', display_image)
                    
                else:
                    # Handle frame drop gracefully (common with RealSense)
                    consecutive_errors += 1
                    if consecutive_errors <= max_consecutive_errors:
                        # Show status even without camera frame
                        status_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        status_msg, color = self._get_sequence_status()
                        cv2.putText(status_frame, status_msg, (50, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                        cv2.putText(status_frame, "Waiting for camera frame...", (50, 280),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
                        cv2.putText(status_frame, f"Direct RealSense, Drops: {consecutive_errors}", (50, 320),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (64, 64, 64), 1)
                        cv2.imshow('Robot Camera Preview', status_frame)
                        
                        # Brief pause to avoid overwhelming the system
                        time.sleep(0.03)
                    else:
                        print(f"Too many consecutive frame errors ({consecutive_errors}), stopping preview")
                        break
                
                # Check for ESC key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("ESC pressed, stopping camera preview")
                    break
                    
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= max_consecutive_errors:
                    if consecutive_errors % 10 == 1:  # Only print every 10th error to avoid spam
                        print(f"Camera frame error (attempt {consecutive_errors}): {e}")
                    time.sleep(0.1)  # Brief pause before retry
                else:
                    print(f"Too many camera errors, stopping preview: {e}")
                    break
                    
        self.running = False


def show_camera_during_robot_sequence(robot_function, pre_delay=3, post_delay=3):
    """
    Show camera preview before, during, and after robot movement.
    
    Args:
        robot_function: Function to execute robot movement
        pre_delay: Seconds to show camera before robot movement (default: 3)
        post_delay: Seconds to show camera after robot movement (default: 3)
    """
    camera = CameraPreview()
    
    try:
        # Start camera preview
        if not camera.start():
            print("Failed to start camera preview")
            return
            
        # Set sequence timing
        camera.set_sequence_timing(pre_delay, post_delay)
        
        # Wait before robot movement with live camera
        print(f"Showing camera preview for {pre_delay} seconds before robot movement...")
        time.sleep(pre_delay)
        
        # Mark robot start and execute robot function
        camera.mark_robot_started()
        print("Starting robot movement...")
        robot_function()
        
        # Mark robot completion and wait after robot movement  
        camera.mark_robot_completed()
        print(f"Showing camera preview for {post_delay} seconds after robot movement...")
        time.sleep(post_delay)
        
    except Exception as e:
        print(f"Error during camera sequence: {e}")
        
    finally:
        # Always stop camera
        camera.stop()


if __name__ == "__main__":
    # Test the managed camera preview
    def test_robot_function():
        print("Robot moving...")
        time.sleep(2)
        print("Robot movement complete")
    
    show_camera_during_robot_sequence(test_robot_function, 3, 3)