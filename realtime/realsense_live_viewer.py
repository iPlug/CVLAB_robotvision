#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2
from bag_inspector import inspect_bag_file
import open3d as o3d

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    
    
    # Check if RGB camera is available
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)
    
    # Configure streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, int(rs.l500_visual_preset.short_range))

    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {depth_scale}")
    
    # Create an align object to align depth frames to color frames
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    pipeline_active = True  # Track pipeline state
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud Viewer', width=800, height=600)
    pcd = None
    try:
        while True:
            if pipeline_active:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                
                # Align the depth frame to color frame
                aligned_frames = align.process(frames)
                
                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue
                
                # Convert images to numpy arrays
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                
                fr = rs.decimation_filter().process(aligned_depth_frame)
                # Create pointcloud object (do this once, outside the loop)
                pc = rs.pointcloud()

                # Inside your loop, after getting aligned frames:
                if not aligned_depth_frame or not color_frame:
                    continue

                # Map pointcloud to color frame for texture mapping  
                pc.map_to(color_frame)

                # Calculate 3D points from depth frame
                points = pc.calculate(fr)

                # Get vertices and convert to numpy array
                v = points.get_vertices()
                verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz coordinates

                # Create Open3D point cloud
                if pcd is None:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(verts)

                    # Optional: Add colors if you want colored point cloud
                    colors = np.asanyarray(color_frame.get_data()).reshape(-1, 3) / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                    vis.add_geometry(pcd)
                else:
                    pcd.points = o3d.utility.Vector3dVector(verts)
                    colors = np.asanyarray(color_frame.get_data()).reshape(-1, 3) / 255.0
                    pcd.colors = o3d.utility.Vector3dVector(colors)

                # Visualize
                if len(verts) > 0:
                    print(f"Extracted {len(verts)} 3D points")
                    
                    # create a visualizer
                    
                    
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                    # vis.run()


                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Stack both images horizontally
                images = np.hstack((color_image, depth_colormap))
                
                # Show images
                cv2.namedWindow('RealSense Live View', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense Live View', images)
            else:
                # Pipeline is stopped, just wait for key press
                cv2.waitKey(30)
            
            # Press 'q' to quit, 'r' to start recording
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r') and pipeline_active:
                print("Starting recording...")
                pipeline_active = record_bag_file(pipeline, config)
                if pipeline_active:
                    print("Recording finished. Resuming live view...")
                else:
                    print("Error during recording. Exiting...")
                    break
    
    finally:
        # Stop streaming only if pipeline is active
        try:
            if pipeline_active:
                pipeline.stop()
        except:
            pass
        cv2.destroyAllWindows()



def record_bag_file(pipeline, config):
    """Record frames to a bag file and return pipeline status"""
    try:
        # Stop current pipeline
        pipeline.stop()
        
        # Create new config for recording
        record_config = rs.config()
        record_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        record_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        record_config.enable_record_to_file('recorded_session.bag')
        
        # Start recording
        pipeline.start(record_config)
        print("Recording started! Press 's' to stop recording...")
        
        recording = True
        while recording:
            try:
                frames = pipeline.wait_for_frames()
                
                # Get color and depth frames for display
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                # Add recording indicator
                cv2.putText(color_image, 'RECORDING...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(color_image, 'Press S to stop', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display
                images = np.hstack((color_image, depth_colormap))
                cv2.imshow('RealSense Live View', images)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    recording = False
                elif key == ord('q'):
                    recording = False
                    pipeline.stop()
                    print("Recording stopped! File saved as 'recorded_session.bag'")
                    return False  # Quit requested
            
            except Exception as e:
                print(f"Error during recording: {e}")
                recording = False
        
        # Stop recording
        pipeline.stop()
        print("Recording stopped! File saved as 'recorded_session.bag'")
        
        # Restart live pipeline
        pipeline.start(config)
        return True  # Pipeline restarted successfully
        
    except Exception as e:
        print(f"Error in recording function: {e}")
        try:
            pipeline.stop()
        except:
            pass
        
        # Try to restart live pipeline
        try:
            pipeline.start(config)
            return True
        except Exception as e2:
            print(f"Could not restart live pipeline: {e2}")
            return False

if __name__ == "__main__":
    main()