#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
from bag_inspector import inspect_bag_file

def main():
    parser = argparse.ArgumentParser(description='RealSense Bag File Viewer')
    parser.add_argument('--input', '-i', type=str, default='recorded_session.bag',
                        help='Path to the bag file')
    parser.add_argument('--realtime', '-rt', action='store_true',
                        help='Play at real-time speed (default: process all frames without timing)')
    parser.add_argument('--inspect', action='store_true',
                        help='Only inspect the bag file without playing')
    args = parser.parse_args()
    
    # Check if bag file exists
    if not os.path.exists(args.input):
        print(f"Bag file '{args.input}' not found!")
        print("Please record a bag file first or specify an existing file with --input")
        return
    
    # Inspect bag file first
    available_streams = inspect_bag_file(args.input)
    
    if args.inspect:
        return
    
    if not available_streams:
        print("Could not determine available streams. Exiting.")
        return
    
    # Create pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Tell config that we will use a recorded device from file to be used by the pipeline through playback
    rs.config.enable_device_from_file(config, args.input)
    
    # Configure streams based on what's available in the bag file
    has_depth = False
    has_color = False
    
    for stream_info in available_streams:
        if stream_info['stream'] == rs.stream.depth and 'width' in stream_info:
            print(f"Enabling depth stream: {stream_info['width']}x{stream_info['height']} @ {stream_info['fps']}fps")
            config.enable_stream(rs.stream.depth, stream_info['width'], stream_info['height'], 
                               stream_info['format'], stream_info['fps'])
            has_depth = True
        elif stream_info['stream'] == rs.stream.color and 'width' in stream_info:
            print(f"Enabling color stream: {stream_info['width']}x{stream_info['height']} @ {stream_info['fps']}fps")
            config.enable_stream(rs.stream.color, stream_info['width'], stream_info['height'], 
                               stream_info['format'], stream_info['fps'])
            has_color = True
    
    if not has_depth and not has_color:
        print("No depth or color streams found in bag file!")
        return
    
    # Start streaming from file
    try:
        profile = pipeline.start(config)
    except Exception as e:
        print(f"Error starting pipeline: {e}")
        print("Trying with automatic stream configuration...")
        
        # Try with automatic configuration
        config = rs.config()
        rs.config.enable_device_from_file(config, args.input)
        try:
            profile = pipeline.start(config)
            print("Successfully started with automatic configuration")
        except Exception as e2:
            print(f"Failed with automatic configuration too: {e2}")
            return
    
    # Get the playback device
    playback = profile.get_device().as_playback()
    
    # Set real-time mode based on argument
    playback.set_real_time(args.realtime)
    
    if not args.realtime:
        print("Non-real-time mode: Processing all frames without timing constraints")
    else:
        print("Real-time mode: Playing at recorded speed")
    
    # Get playback duration
    try:
        duration = playback.get_duration()
        print(f"Bag file duration: {duration.total_seconds():.2f} seconds")
    except:
        print("Could not determine bag file duration")
    
    # Create alignment primitive only if we have both color and depth
    align = None
    if has_color and has_depth:
        align = rs.align(rs.stream.color)
        print("Frame alignment enabled (depth to color)")
    
    frame_count = 0
    
    try:
        while True:
            # Get frames - use poll_for_frames for non-blocking operation
            frames_available, frames = pipeline.try_wait_for_frames(timeout_ms=100)
            
            if not frames_available:
                print(f"End of file reached. Processed {frame_count} frames.")
                break
            
            frame_count += 1
            
            # Process frames based on what's available
            depth_image = None
            color_image = None
            
            if align and has_color and has_depth:
                # Align frames if both are available
                aligned_frames = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if aligned_depth_frame:
                    depth_image = np.asanyarray(aligned_depth_frame.get_data())
                if color_frame:
                    color_image = np.asanyarray(color_frame.get_data())
            else:
                # Process individual streams
                if has_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        depth_image = np.asanyarray(depth_frame.get_data())
                
                if has_color:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        color_image = np.asanyarray(color_frame.get_data())
            
            # Create visualization
            display_images = []
            
            if color_image is not None:
                # Add frame info to color image
                info_image = color_image.copy()
                cv2.putText(info_image, f'Frame: {frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                timestamp = frames.get_timestamp()
                cv2.putText(info_image, f'Time: {timestamp:.2f}ms', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                display_images.append(info_image)
            
            if depth_image is not None:
                # Apply colormap on depth image
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                display_images.append(depth_colormap)
            
            if not display_images:
                continue
            
            # Combine images for display
            if len(display_images) == 1:
                combined_image = display_images[0]
            else:
                # Resize images to same height if needed
                max_height = max(img.shape[0] for img in display_images)
                resized_images = []
                for img in display_images:
                    if img.shape[0] != max_height:
                        aspect_ratio = img.shape[1] / img.shape[0]
                        new_width = int(max_height * aspect_ratio)
                        img = cv2.resize(img, (new_width, max_height))
                    resized_images.append(img)
                combined_image = np.hstack(resized_images)
            
            # Display images
            cv2.namedWindow('RealSense Bag Playback', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense Bag Playback', combined_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space to pause/resume
                print("Paused. Press space again to resume, 'q' to quit.")
                playback.pause()
                while True:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(' '):
                        playback.resume()
                        print("Resumed playback")
                        break
                    elif key == ord('q'):
                        return
            elif key == ord('s'):  # Save current frame
                save_frame(color_image if color_image is not None else None, 
                          depth_image if depth_image is not None else None, 
                          frame_count)
    
    except Exception as e:
        print(f"Error during playback: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"Playback finished. Total frames processed: {frame_count}")

def save_frame(color_image, depth_image, frame_number):
    """Save current color and depth frames"""
    saved_files = []
    
    if color_image is not None:
        color_filename = f'frame_{frame_number:06d}_color.png'
        cv2.imwrite(color_filename, color_image)
        saved_files.append(color_filename)
    
    if depth_image is not None:
        depth_filename = f'frame_{frame_number:06d}_depth.png'
        # Convert depth to 16-bit for saving
        cv2.imwrite(depth_filename, depth_image.astype(np.uint16))
        saved_files.append(depth_filename)
    
    if saved_files:
        print(f"Saved frames: {', '.join(saved_files)}")
    else:
        print("No frames to save")

if __name__ == "__main__":
    main()