## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.
#####################################################
##           Continuous Export to PLY              ##
#####################################################
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

# Create directory for saving PLY files if it doesn't exist
os.makedirs("captured_clouds", exist_ok=True)

# Declare pointcloud object, for calculating pointclouds and texture mappings
pc = rs.pointcloud()
# We want the points object to be persistent so we can display the last cloud when a frame drops
points = rs.points()
# Declare RealSense pipeline, encapsulating the actual device and sensors
pipe = rs.pipeline()
config = rs.config()

# Enable depth stream and color stream
config.enable_stream(rs.stream.depth, rs.format.z16, 30)
config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

# Start streaming with chosen configuration
pipe.start(config)
profile = pipe.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 5)
# We'll use the colorizer to generate texture for our PLY
colorizer = rs.colorizer()


# Configure PLY saving settings
def create_ply_saver(filename):
    ply = rs.save_to_ply(filename)
    # Set options to the desired values - textual PLY with normals
    ply.set_option(rs.save_to_ply.option_ply_binary, False)
    ply.set_option(rs.save_to_ply.option_ply_normals, True)
    return ply


# Add variables for frame capture timing
frame_count = 0
max_frames = 30  # Capture 30 frames total
capture_interval = 0.5  # Capture every 0.5 seconds
last_capture_time = 0

try:
    print("Capturing point clouds continuously. Press 'q' or ESC to exit...")

    while True:
        # Get current time for interval checking
        current_time = time.time()

        # Wait for the next set of frames from the camera
        frames = pipe.wait_for_frames()

        # Process the frames with the colorizer
        colorized = colorizer.process(frames)

        # Get color frame for display
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert color frame to numpy array for OpenCV
        color_image = np.asanyarray(color_frame.get_data())

        # Check if it's time to capture a frame
        if current_time - last_capture_time >= capture_interval:
            # Create filename for this capture
            ply_filename = f"captured_clouds/cloud_{frame_count:03d}.ply"

            # Create ply saver with our settings
            ply = create_ply_saver(ply_filename)

            print(f"Saving to {ply_filename}...")
            # Apply the processing block to the frameset which contains the depth frame and the texture
            ply.process(colorized)
            print(f"Saved point cloud {frame_count+1}/{max_frames}")

            # Update counters
            frame_count += 1
            last_capture_time = current_time

            # Check if we've captured all frames
            if frame_count >= max_frames:
                print("Captured all frames. Exiting...")
                break

        # Display the color image
        cv2.imshow("RealSense", color_image)

        # Check for exit key
        key = cv2.waitKey(1)
        if key in (27, ord("q")):
            break

finally:
    # Stop streaming
    pipe.stop()
    # Close any open windows
    cv2.destroyAllWindows()
    print("Program finished.")
