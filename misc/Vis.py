import cv2
import numpy as np
import pyrealsense2 as rs


def visualize_realsense_l515(fps=15, preset="SHORT_RANGE", resolution=(640, 480)):
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    # Start streaming
    profile = config.resolve(pipeline)
    pipeline.start(config)

    # Get depth sensor
    depth_sensor = profile.get_device().first_depth_sensor()

    # Available presets for L515
    presets = {
        "DEFAULT": 0,
        "CUSTOM": 1,
        "LOW_AMBIENT": 2,
        "MAX_RANGE": 3,
        "NO_AMBIENT": 4,
        "SHORT_RANGE": 5,
    }

    # Set visual preset
    if preset in presets:
        depth_sensor.set_option(rs.option.visual_preset, presets[preset])
        print(f"Set visual preset to: {preset}")
    else:
        print(f"Invalid preset '{preset}'. Using DEFAULT.")
        depth_sensor.set_option(rs.option.visual_preset, 0)

    # Create colorizer for better depth visualization
    colorizer = rs.colorizer()

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()

            # Get depth and color frames
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Stack the images horizontally
            images = np.hstack((color_image, depth_image))

            # Display the images
            cv2.namedWindow("RealSense L515", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense L515", images)

            # Break loop on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")


if __name__ == "__main__":
    # Example usage with different parameters
    # Available presets: DEFAULT, CUSTOM, LOW_AMBIENT, MAX_RANGE, NO_AMBIENT, SHORT_RANGE
    # Available fps: 6, 15, 30 (depending on resolution)
    # Available resolutions: (1024, 768), (640, 480), (320, 240)

    visualize_realsense_l515(
        fps=15,  # Lower fps than default
        preset="SHORT_RANGE",  # Choose appropriate preset for your environment
        resolution=(640, 480),  # Resolution
    )
