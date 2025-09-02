#!/usr/bin/env python3

import pyrealsense2 as rs
import argparse
import os
import sys

def inspect_bag_file(bag_path):
    """Inspect bag file to determine available streams"""
    print(f"Inspecting bag file: {bag_path}")
    
    # Create a temporary pipeline to inspect the bag
    temp_pipeline = rs.pipeline()
    temp_config = rs.config()
    rs.config.enable_device_from_file(temp_config, bag_path)
    
    try:
        # Start without specifying streams to see what's available
        temp_profile = temp_pipeline.start(temp_config)
        device = temp_profile.get_device()
        
        print("Available streams in bag file:")
        available_streams = []
        
        # Get all available streams
        for i, sensor in enumerate(device.sensors):
            print(f"  Sensor {i}: {sensor.get_info(rs.camera_info.name)}")
            for j, stream_profile in enumerate(sensor.get_stream_profiles()):
                stream = stream_profile.stream_type()
                format_type = stream_profile.format()
                
                if stream_profile.is_video_stream_profile():
                    video_profile = stream_profile.as_video_stream_profile()
                    width = video_profile.width()
                    height = video_profile.height()
                    fps = video_profile.fps()
                    print(f"    Stream {j}: {stream} {width}x{height} @ {fps}fps, format: {format_type}")
                    available_streams.append({
                        'stream': stream,
                        'width': width,
                        'height': height,
                        'format': format_type,
                        'fps': fps
                    })
                else:
                    print(f"    Stream {j}: {stream}, format: {format_type}")
                    available_streams.append({
                        'stream': stream,
                        'format': format_type
                    })
        
        temp_pipeline.stop()
        return available_streams
        
    except Exception as e:
        print(f"Error inspecting bag file: {e}")
        temp_pipeline.stop()
        return None

def count_frames(pipeline, playback):
    """Estimate frame count by quickly iterating through the bag"""
    print("  Counting frames... (this may take a moment)")
    
    # Reset to beginning
    playback.seek(rs.time(0))
    
    frame_count = 0
    last_reported = 0
    
    try:
        while True:
            frames_available, frames = pipeline.try_wait_for_frames(timeout_ms=10)
            
            if not frames_available:
                break
            
            frame_count += 1
            
            # Report progress every 1000 frames
            if frame_count - last_reported >= 1000:
                print(f"    Counted {frame_count} frames so far...")
                last_reported = frame_count
            
            # Safety break for very large files
            if frame_count > 100000:
                print(f"    Stopped counting at {frame_count} frames (file is very large)")
                break
                
    except Exception as e:
        print(f"    Frame counting stopped due to: {e}")
    
    return frame_count

def main():
    parser = argparse.ArgumentParser(description='RealSense Bag File Inspector')
    parser.add_argument('bag_file', help='Path to the bag file to inspect')
    args = parser.parse_args()
    
    if not inspect_bag_file(args.bag_file):
        sys.exit(1)
    
    print("\n" + "="*60)
    print("INSPECTION COMPLETE")
    print("="*60)
    print("Use the configuration suggestions above in your Python code.")
    print("If you're still having issues, the bag file might be corrupted")
    print("or from an incompatible RealSense SDK version.")

if __name__ == "__main__":
    main()
