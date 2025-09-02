# Eye-in-Hand Configuration Setup

This document explains how to use the eye-in-hand configuration where the camera is mounted on the robot's end effector.

## Overview

The eye-in-hand setup differs from the original hand-to-eye calibration in several key ways:

### Hand-to-Eye (Original System)
- Camera is statically positioned
- Robot moves to point at objects detected by camera
- Requires camera-robot transformation matrix from calibration
- Uses `modular_track_object.py`

### Eye-in-Hand (New System) 
- Camera is mounted on robot end effector
- Camera moves with robot
- No coordinate transformation needed (camera and robot share same reference frame)
- Uses `eye_in_hand_tracker.py`

## Key Features

### 🎯 Three Operating Modes
- **OBSERVE**: Maintain safe viewing distance from objects (350mm default, >250mm lidar minimum)
- **APPROACH**: Move closer to objects for manipulation (280mm default, respects lidar minimum)  
- **MANIPULATE**: Precise positioning at lidar effective range (250mm minimum)

### 🔄 Real-time Strategy Switching
- **BEV Clustering**: Fast bird's-eye view processing (30+ FPS)
- **DBSCAN Clustering**: High-quality 3D point cloud clustering (5 FPS)
- Switch strategies runtime with keyboard controls

### 🤖 LOOK_AT_TABLE Integration
- Robot positions to `[0, 0, 0, -45, 0, 45]` for workspace viewing
- Camera mounted on end effector provides direct workspace observation  
- Respects 250mm minimum lidar effective range for all positioning

### 📏 Lidar Range Constraints
- **Minimum distance**: 250mm (lidar effective range)
- **All robot positioning**: Maintains >250mm from detected objects
- **Calibration positions**: Designed to respect minimum range
- **Safety margins**: Additional buffer above minimum for reliable operation

## Installation & Setup

### 1. Hardware Setup
```bash
# Mount camera on robot end effector
# Ensure stable mounting with minimal vibration
# Connect camera USB to computer
# Connect robot to computer via USB/serial
```

### 2. Software Requirements
```bash
# Same requirements as existing system
python -m pip install -r requirements.txt
pip install pyrealsense2==2.54.2.5684
```

### 3. Eye-in-Hand Calibration Required
Unlike hand-to-eye setup, **eye-in-hand calibration is needed** to determine the transformation between camera frame and robot end effector frame.

```bash
# Run eye-in-hand calibration (5 positions, standard)
python eye_in_hand_calibration.py --positions 5 --output eye_in_hand_transform.npy

# Quick calibration (3 positions)
python eye_in_hand_calibration.py --positions 3

# Precision calibration (7+ positions)  
python eye_in_hand_calibration.py --positions 7
```

## Usage

### Basic Operation
```bash
# Real-time eye-in-hand tracking (recommended)
python eye_in_hand_tracker.py --config eye_in_hand_realtime

# High-precision tracking (slower but more accurate)
python eye_in_hand_tracker.py --config eye_in_hand_precision

# Start with BEV strategy (default)
python eye_in_hand_tracker.py --strategy bev

# Start with DBSCAN strategy  
python eye_in_hand_tracker.py --strategy dbscan

# Use specific transformation matrix
python eye_in_hand_tracker.py --transform my_eye_in_hand_transform.npy
```

### Keyboard Controls

#### Basic Controls
- `q` - Quit application
- `h` - Toggle help display
- `p` - Toggle performance info
- `space` - Pause/Resume tracking

#### Mode Switching
- `o` - Switch to **OBSERVE** mode (maintain viewing distance)
- `a` - Switch to **APPROACH** mode (move closer to objects)
- `m` - Switch to **MANIPULATE** mode (precise positioning)

#### Strategy Switching
- `1` - Switch to BEV clustering strategy (fast)
- `2` - Switch to DBSCAN clustering strategy (precise)

#### Robot Control
- `l` - **LOOK_AT_TABLE** (position robot to view workspace)
- `s` - Stop robot movement immediately
- `r` - Home position (if available)

## Configuration Options

### Eye-in-Hand Realtime (`eye_in_hand_realtime`)
```python
{
    'clustering_params': {
        'min_cluster_size': 15,      # Smaller for closer objects
        'max_cluster_size': 80,      # Adjusted for typical sizes
        'morphology_kernel_size': 3,
        'window_size': 3             # Fast response
    },
    'tracking_params': {
        'update_interval': 0.8,      # Faster updates (800ms)
        'fast_update_interval': 0.3, # Very fast when needed (300ms)
        'adaptive_timing': True,
        'max_no_object_time': 3.0
    },
    'robot_params': {
        'min_height': 150,           # Higher minimum for safety
        'max_speed': 60,             # Slower for precise positioning
        'approach_distance': 100,    # Distance for approach mode
        'retreat_distance': 200      # Distance for observe mode
    }
}
```

### Eye-in-Hand Precision (`eye_in_hand_precision`)
```python
{
    'clustering_params': {
        'min_cluster_size': 10,      # Even smaller clusters
        'max_cluster_size': 150,     # Larger range
        'morphology_kernel_size': 5,
        'window_size': 5             # More temporal filtering
    },
    'tracking_params': {
        'update_interval': 1.5,      # Slower updates for precision
        'fast_update_interval': 0.8,
        'adaptive_timing': True,
        'max_no_object_time': 8.0
    },
    'robot_params': {
        'min_height': 180,           # Higher safety margin
        'max_speed': 30,             # Very slow for precision
        'approach_distance': 80,     # Closer approach
        'retreat_distance': 250      # Further retreat
    }
}
```

## Workflow Examples

### 1. Object Inspection Workflow
```bash
# Start system
python eye_in_hand_tracker.py --config eye_in_hand_realtime

# 1. Position robot to view workspace
Press 'l' (LOOK_AT_TABLE)

# 2. Switch to observe mode to survey objects
Press 'o' (OBSERVE mode)

# 3. Switch to approach mode for closer inspection
Press 'a' (APPROACH mode)

# 4. Switch to manipulate mode for interaction
Press 'm' (MANIPULATE mode)
```

### 2. Precision Manipulation Workflow
```bash
# Start with precision configuration
python eye_in_hand_tracker.py --config eye_in_hand_precision

# 1. Initial workspace view
Press 'l' (LOOK_AT_TABLE)

# 2. High-precision strategy for fine details
Press '2' (DBSCAN strategy)

# 3. Approach target object
Press 'a' (APPROACH mode)

# 4. Fine positioning for manipulation
Press 'm' (MANIPULATE mode)
```

### 3. Real-time Monitoring Workflow
```bash
# Fast real-time tracking
python eye_in_hand_tracker.py --config eye_in_hand_realtime --strategy bev

# 1. Fast BEV processing for real-time response
Press '1' (BEV strategy)

# 2. Continuous observation mode
Press 'o' (OBSERVE mode)

# 3. Quick approach when needed
Press 'a' (APPROACH mode)
```

## Technical Details

### Coordinate System
- **Camera coordinates**: Objects detected relative to camera/end-effector
- **Robot coordinates**: Direct control of end-effector position
- **No transformation**: Camera and robot share reference frame

### Distance Calculations
```python
# Object distance from end-effector
distance = sqrt(x² + y² + z²)

# Closest object selection
closest = min(objects, key=lambda obj: distance(obj.position))
```

### Movement Strategy
```python
# Relative movement calculation
target_position = current_position + desired_offset

# Mode-specific offsets:
observe_offset = [0, 0, retreat_distance]    # 200mm back
approach_offset = [0, 0, approach_distance]  # 100mm forward  
manipulate_offset = [0, 0, 50]               # 50mm for manipulation
```

## Safety Features

### Workspace Constraints
- **Minimum height**: 150mm (realtime) / 180mm (precision)
- **Maximum speed**: 60 (realtime) / 30 (precision)
- **Rate limiting**: Prevents command flooding
- **Emergency stop**: 's' key for immediate halt

### Collision Avoidance
- **Approach limits**: Configurable minimum distances
- **Workspace bounds**: Prevents out-of-reach movements
- **Timeout protection**: Commands auto-expire

## Performance Characteristics

### Real-time Configuration
- **Detection FPS**: 30+ fps with BEV strategy
- **Robot response**: 300-800ms update intervals
- **Memory usage**: ~200MB typical

### Precision Configuration  
- **Detection FPS**: 5-10 fps with DBSCAN strategy
- **Robot response**: 800-1500ms update intervals
- **Memory usage**: ~500MB typical

## Troubleshooting

### Common Issues

#### 1. Camera Not Detected
```bash
# Check camera connection
lsusb | grep Intel

# Restart RealSense service
sudo systemctl restart realsense
```

#### 2. Robot Not Responding
```bash
# Check robot connection
ls /dev/ttyUSB*

# Test robot connection
from module.robot.mycobot_controller import MyCobotController
robot = MyCobotController()
robot.connect()
```

#### 3. Poor Detection Performance
```bash
# Switch to faster strategy
Press '1' for BEV clustering

# Adjust lighting conditions
# Clean camera lens
# Reduce background clutter
```

#### 4. Robot Movement Too Slow/Fast
- Edit `robot_params.max_speed` in configuration
- Use precision config for slower movement
- Use realtime config for faster movement

### Debug Mode
```bash
# Enable debug logging
python eye_in_hand_tracker.py --config eye_in_hand_realtime --verbose

# Check configuration
python test_eye_in_hand.py
```

## Validation

### Test System
```bash
# Run comprehensive tests
python test_eye_in_hand.py

# Expected output:
# TEST RESULTS: 7/7 tests passed
# + All eye-in-hand tests PASSED!
```

### Test Coverage
- ✓ Configuration loading
- ✓ Mode switching
- ✓ Coordinate extraction
- ✓ Relative movement calculation
- ✓ Object distance calculation
- ✓ Mode-specific behaviors
- ✓ Configuration presets

## Migration from Hand-to-Eye

### Key Differences
| Feature | Hand-to-Eye | Eye-in-Hand |
|---------|-------------|-------------|
| Camera position | Static | Moves with robot |
| Calibration | Required | Not needed |
| Coordinate transform | Camera→Robot | Direct control |
| Application | `modular_track_object.py` | `eye_in_hand_tracker.py` |
| Control strategy | Point at objects | Approach objects |

### Migration Steps
1. **Physical setup**: Mount camera on end effector
2. **Software**: Use `eye_in_hand_tracker.py` instead of `modular_track_object.py`  
3. **Configuration**: Use eye-in-hand specific configs
4. **Operation**: Use LOOK_AT_TABLE to position robot for workspace viewing

## Advanced Usage

### Custom Configurations
```python
# Create custom eye-in-hand configuration
from eye_in_hand_tracker import EyeInHandTracker

tracker = EyeInHandTracker()
tracker.approach_distance = 75    # Custom approach distance
tracker.current_robot_mode = "manipulate"  # Start in manipulate mode
```

### Integration with Other Systems
```python
# Use as a component in larger system
from eye_in_hand_tracker import EyeInHandTracker

class ManipulationSystem:
    def __init__(self):
        self.tracker = EyeInHandTracker(config_name='eye_in_hand_precision')
        
    def approach_object(self):
        self.tracker.current_robot_mode = "approach"
        
    def manipulate_object(self):
        self.tracker.current_robot_mode = "manipulate"
```

The eye-in-hand configuration provides a more direct and intuitive approach to robot vision, eliminating the need for complex calibration while enabling precise object manipulation through camera feedback.