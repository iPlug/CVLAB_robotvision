# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This is a modern Python-based computer vision and robotics project with a **modular architecture**. Setup requires:

1. **Virtual Environment**: Use Python 3.8.0 with the provided `.venv`
2. **Dependencies**: `pip install -r requirements.txt`
3. **OpenCV**: `pip install opencv-python==4.11.0.*` (**Critical**: Version 4.11.0+ required for CharUco coordinate system compatibility)
4. **Robot Control**: `pip install pymycobot` (for myCobot 280 integration)
5. **RealSense SDK**: `pip install pyrealsense2==2.54.2.5684`
6. **Scientific Computing**: `pip install scipy` (for coordinate transformations and rotations)
7. **PyTorch** (for PointNet models): `pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`

**Python Interpreter:** Always use `.venv/Scripts/python.exe` for all testing and script execution.

**Important Version Notes:**
- **OpenCV 4.11.0**: Required for proper CharUco coordinate system handling (Z-axis direction changed in 4.6.0+)
- **pymycobot**: Essential for myCobot 280 robot control and coordinate system integration
- **scipy**: Used for robust coordinate transformations and Rotation matrix calculations

## Project Architecture

This project has evolved into a **modern modular system** with a clean separation between legacy research components and production-ready modular applications.

### Modern Modular Architecture (Primary System)

```
module/
├── core/                    # Core interfaces and orchestration
│   ├── object_detector.py   # Main orchestration class
│   └── visualization_engine.py  # Unified visualization system
├── strategies/              # Pluggable detection algorithms
│   ├── bev_clustering_strategy.py      # Real-time BEV clustering
│   ├── dbscan_clustering_strategy.py   # 3D point cloud clustering
│   ├── charuco_detection_strategy.py   # CharUco pattern detection for calibration
│   └── pointnet_subspace_classifier.py # Neural network classification
├── processing/              # Data processing utilities
│   ├── point_cloud_processor.py # ROI filtering, downsampling
│   ├── image_processor.py       # BEV generation, morphology
│   └── temporal_filter.py       # Multi-frame stability
├── sensors/                 # Hardware abstraction layer
│   ├── realsense_manager.py     # Intel RealSense cameras
│   └── file_based_manager.py    # Bag file replay
├── robot/                   # Advanced robot control
│   ├── mycobot_controller.py    # Enhanced robot control
│   └── command_queue.py         # Async command system
├── config/                  # Configuration management
│   └── config_factory.py       # 14 predefined presets
└── apps/                    # High-level applications
    ├── calibration_app.py       # Modular calibration system
    ├── charuco_calibration_app.py # CharUco-based calibration system
    └── unified_detection_app.py # Research platform
```

### Key Design Patterns
- **Strategy Pattern**: Pluggable detection algorithms (BEV ↔ DBSCAN ↔ PointNet)
- **Factory Pattern**: Configuration management with presets
- **Observer Pattern**: Async robot command system
- **Template Method**: Unified processing pipeline

## Main Applications

### Modular Applications (Recommended)

#### **Real-time Object Tracking**
```bash
# Start with real-time BEV tracking (30+ FPS)
python modular_track_object.py --config realtime_tracking

# High-precision tracking (slower, more accurate)  
python modular_track_object.py --config precision_tracking

# Adaptive tracking with strategy switching
python modular_track_object.py --config adaptive_tracking
```

**Features:**
- **30+ FPS performance** with BEV clustering
- **Runtime strategy switching** (BEV ↔ DBSCAN) via keyboard ('s')
- **Async robot control** - visualization never blocks during robot movement
- **3-panel visualization**: Color | BEV | Point Cloud views
- **Performance monitoring** with adaptive timing

#### **Camera-Robot Calibration**

##### **CharUco-Based Calibration (Primary Method)**
```bash
# Standard eye-in-hand calibration with 8x11 CharUco pattern
python charuco_eye_in_hand.py

# Quick calibration with fewer positions
python charuco_eye_in_hand.py --config charuco_quick

# High-precision calibration with more positions
python charuco_eye_in_hand.py --config charuco_precise

# Custom pattern size and position
python charuco_eye_in_hand.py --pattern 8 11 --pattern-position 400 0 0

# Real-time position prediction using calibrated transformation
python charuco_predict_position.py --verbose

# Use center origin coordinate system (simple offset method)
python charuco_predict_position.py --center-origin
```

**CharUco Features:**
- **High-precision calibration** using 7x5 CharUco patterns (39mm squares, 29mm markers)
- **Eye-in-hand setup** with robot-mounted camera observing fixed pattern
- **Sub-pixel corner detection** with ArUco marker validation
- **OpenCV 4.11.0 compatible** coordinate system handling with 90° Z-rotation correction
- **Automatic unit conversion** between mm/meter coordinate systems  
- **Real-time position prediction** with transformation matrix validation
- **Fixed coordinate system alignment** - CharUco axes now properly align with robot axes
- **Comprehensive logging** with detailed pose data and quality scores  
- **Resolved 95° systematic rotation error** through coordinate system correction
- **Simplified center origin** - uses simple offset method instead of complex coordinate transformations

##### **Real-time CharUco Calibration (Alternative)**
```bash
# Real-time calibration with automatic robot movement
cd charuco-pose-estimation
python charuco_realtime_pose.py

# Custom CharUco pattern size
python charuco_realtime_pose.py --square-size 0.025 --marker-size 0.02 --board-size 8 11
```

**Real-time Features:**
- **Automatic robot positioning** via 'c' key (cycles through preset poses)
- **Live CharUco detection** with real-time pose visualization  
- **Interactive calibration point collection** via 's' key
- **Advanced validation** with known pattern position verification
- **22mm position accuracy** and **8° rotation accuracy** achieved with coordinate system correction
- **Factory RealSense intrinsics** automatically extracted from camera
- **Compatible with L515 and other RealSense cameras**
- **Fixed 95° systematic rotation error** through 90° Z-axis coordinate correction

**Controls:**
- **'c'**: Move robot to next preset calibration position
- **'s'**: Collect calibration point (robot + CharUco poses)
- **'q'**: Calculate eye-in-hand transformation and validate

##### **Real-time CharUco Hand Position Prediction**
```bash
# Standard real-time position prediction relative to robot hand
python charuco_realtime_hand_position.py

# Use custom transformation matrix file
python charuco_realtime_hand_position.py --transform-matrix my_calibration.npy

# Custom CharUco pattern parameters
python charuco_realtime_hand_position.py --pattern 8 11 --square-size 25.0 --marker-size 20.0

# Verbose output with detailed position metrics
python charuco_realtime_hand_position.py --verbose
```

**Hand Position Features:**
- **Real-time position prediction** of CharUco patterns relative to robot hand coordinate system
- **Live coordinate transformation** using pre-calibrated eye-in-hand transformation matrix
- **Robot coordinate system display** (X=forward, Y=left, Z=up) with color-coded axes
- **Position stability metrics** with standard deviation tracking over time
- **Performance monitoring** with FPS and detection rate statistics
- **Robot integration** shows current robot hand position when connected
- **Interactive controls** for resetting statistics and toggling verbose mode
- **Visual feedback** with 3-panel display showing position, rotation, and stability metrics

**Controls:**
- **'q' or ESC**: Quit the application
- **'r'**: Reset position tracking statistics
- **'v'**: Toggle verbose output mode showing detailed frame-by-frame information

**Use Cases:**
- **Visual servoing** - Real-time feedback for robot positioning relative to objects
- **Pick and place operations** - Precise object localization before grasping
- **Quality inspection** - Monitoring position accuracy and stability over time
- **Calibration validation** - Testing eye-in-hand transformation accuracy in real scenarios
- **Research and development** - Analyzing coordinate transformation performance

##### **Legacy Point-Based Calibration**
```bash
# Standard calibration (5 points, balanced)
python modular_calibrate_camera.py

# High-precision calibration (7 points, slow)
python modular_calibrate_camera.py --config calibration_precise

# Quick calibration (3 points, fast)
python modular_calibrate_camera.py --config calibration_quick

# List available configurations
python modular_calibrate_camera.py --list-configs
```

**Legacy Features:**
- **3 calibration presets** with different speed/accuracy tradeoffs
- **Async robot movement** for smooth UI during calibration
- **Real-time object detection** with automatic object selection
- **Validation testing** with configurable accuracy thresholds
- **Comprehensive logging** and error reporting

#### **Research Platform**
```bash
# Unified detection app with runtime strategy switching
python module/apps/unified_detection_app.py
```

**Features:**
- **Runtime strategy switching** without restart
- **Classification toggle** (detection only vs detection + classification)
- **Performance comparison** between strategies
- **Comprehensive visualization** and metrics

### Legacy Applications (Still Functional)

#### **Original Tracking System**
```bash
python track_object.py  # Basic tracking with blocking robot commands
```

#### **Original Calibration System** 
```bash
python calibrate_camera.py  # Basic calibration without async features
```

#### **Research Pipeline (PointNet Focus)**
```bash
# Sequential processing for PointNet training/evaluation
python 0_bag_extractor.py      # Extract data from RealSense bags
python 1_remove_background.py  # Background removal  
python 2_copy_data_folder.py   # Organize for training
python 3_extract_features.py   # PointNet feature extraction
python 4_test_classification.py # Classification testing
python 5_folder_recognition.py  # Batch recognition
python 6_grab_points.py        # Point cloud utilities
```

## Configuration System

The project uses a **configuration factory pattern** with 14 predefined presets:

### Real-time Configurations
- **`realtime_tracking`**: 30+ FPS BEV clustering with robot integration
- **`realtime_bev`**: Fast BEV clustering for general detection
- **`fast_pointnet`**: Optimized PointNet for real-time classification

### High-Quality Configurations  
- **`precision_tracking`**: Slower, more accurate tracking
- **`high_quality_bev`**: Enhanced BEV with larger images and temporal filtering
- **`offline_pointnet`**: Full-quality PointNet for research

### Calibration Configurations

#### **CharUco Calibration (Primary)**
- **`charuco_standard`**: 8-12 positions, balanced speed/accuracy (recommended)
- **`charuco_precise`**: 15+ positions, high precision, slower
- **`charuco_quick`**: 8 positions, fast calibration
- **Pattern**: 7x5 CharUco (39mm squares, 29mm markers)
- **Dictionary**: DICT_4X4_50 ArUco markers

#### **Legacy Point-Based Calibration**
- **`calibration_standard`**: 5 points, balanced speed/accuracy
- **`calibration_precise`**: 7 points, high precision, slower
- **`calibration_quick`**: 3 points, fast calibration

### Adaptive Configurations
- **`adaptive_tracking`**: Dynamic strategy switching based on performance
- **`debug_mode`**: Enhanced logging and timing information

### Usage Examples
```python
from module.config.config_factory import ConfigFactory

# Create configuration
config = ConfigFactory.create_preset('realtime_tracking')

# List available presets
presets = ConfigFactory.get_available_presets()

# Create custom hybrid configuration
configs = ConfigFactory.create_hybrid_config(
    bev_params={'clustering_params': {'min_cluster_size': 15}},
    pointnet_params={'subspace_params': {'ndim': 30}}
)
```

## Advanced Features

### Async Robot Control System
- **Non-blocking commands**: Robot movement never freezes visualization
- **Priority queue**: Stop > Point At Object > Home Position  
- **Command callbacks**: Notification when movements complete
- **Safety constraints**: Workspace bounds, height limits, rate limiting
- **Timeout protection**: Commands automatically timeout to prevent hanging

### Multi-Strategy Detection
- **BEV Clustering**: 30+ FPS, real-time optimized, bird's-eye view image processing
- **DBSCAN Clustering**: ~5 FPS, high-quality 3D point cloud clustering  
- **PointNet Classification**: ~1 FPS, neural network object identification
- **Runtime switching**: Change strategies without restart via keyboard commands

### Hardware Abstraction
- **Sensor abstraction**: Works with RealSense cameras or bag file replay
- **Robot abstraction**: myCobot integration with fallback for missing hardware
- **Cross-platform**: Windows/Linux support with identical APIs

### Coordinate Systems

#### **myCobot 280 Robot Coordinates**
- **Origin**: Robot base (physical base of robot)
- **X-axis**: Forward direction (positive away from base)
- **Y-axis**: Side direction (positive to the left when facing forward)
- **Z-axis**: Vertical up (positive upward from base)
- **Units**: Millimeters for position, degrees for orientation
- **Format**: [x, y, z, rx, ry, rz] where rx/ry/rz are Euler angles
- **Working envelope**: 280mm effective radius, 350mm reach

#### **OpenCV CharUco Coordinates (Version 4.11.0)**
- **Pattern origin**: Bottom-left corner or center of CharUco pattern
- **Bottom-left mode**: Origin at bottom-left corner (default, most stable)
- **Center origin mode**: Uses simple offset (+136.5mm X, +97.5mm Y) from bottom-left
- **X-axis**: Along pattern edge (after 90° Z-rotation correction)
- **Y-axis**: Along pattern edge (after 90° Z-rotation correction) 
- **Z-axis**: Out of the board plane (pointing up from table)
- **Units**: Meters (converted from mm pattern sizes)
- **Coordinate correction**: 90° Z-rotation applied to align with robot coordinate system
- **Visual alignment**: GREEN→Robot X, RED→Robot Y, BLUE→Robot Z
- **Simplified approach**: Center origin uses simple offset instead of complex coordinate transformations

#### **RealSense Camera Coordinates**
- **Color frame**: Standard camera coordinates (x=right, y=down, z=forward)
- **Depth frame**: Aligned with color using factory extrinsics
- **IMU alignment**: Gravity-aligned coordinate frame available
- **Factory calibration**: Integrated into transformation chain

#### **Coordinate Transformations**
- **Transformation chain**: Robot → Color → Depth (when needed)
- **Eye-in-hand matrix**: 4x4 homogeneous transformation from robot base to camera
- **Unit conversion**: Automatic handling between mm (robot) and meters (OpenCV)
- **Current accuracy**: ~20mm validation error (real-time system with proper coordinate alignment)

#### **Resolved Issues** ✅
- **OpenCV 4.6+ compatibility**: Z-axis direction corrected with Y-axis flip (reference implementation)  
- **Unit scale problems**: Fixed mm vs meter conversion in transformation pipeline
- **Coordinate frame alignment**: Robot base coordinate system properly aligned and validated
- **myCobot Y-axis convention**: Confirmed +Y = left side (corrected from documentation)
- **Calibration accuracy**: Improved to 20mm validation error with proper coordinate system alignment
- **Real-time calibration**: Added interactive system with automatic robot positioning
- **Point Cloud Visualization Aspect Ratio**: Fixed a persistent issue where the point cloud visualization was stretched into a square instead of maintaining the camera's landscape aspect ratio.
  - **Root Cause**: The 3D-to-2D projection was using the wrong camera intrinsics. The depth stream was aligned to the color stream, meaning the resulting point cloud was in the color camera's coordinate space. However, the projection was incorrectly using the depth camera's intrinsics.
  - **Solution**:
    1.  The `setup_camera` function was updated to explicitly query and store the `color_intrinsics` from the active `pyrealsense2` color stream profile.
    2.  The `_create_lidar_point_cloud_view` function was corrected to use these `color_intrinsics` for the projection calculation.
  - **Key Takeaway**: When using an aligned depth frame from `pyrealsense2`, the color camera's intrinsics must be used for accurate 3D-to-2D projections.

#### **Coordinate System Validation**
- **myCobot coordinates**: +X=forward, +Y=left, +Z=up (confirmed from testing)
- **Pattern position**: [300, 100, 0]mm = 300mm forward, 100mm left, table level
- **Calibration accuracy**: 20mm mean position error (excellent for robotics)
- **Coordinate stability**: ±1mm variance indicating excellent precision
- **Axis orientation**: Red(X)→Forward, Green(Y)→Left, Blue(Z)→Up
- **Origin location**: Bottom-left corner of CharUco pattern (corrected coordinate system)

## Testing Framework

The project includes a **comprehensive testing system** with 100% pass rates:

### Test Execution
```bash
# Test modular calibration system (7 tests)
python test/test_modular_calibration.py

# Test async robot command system (5 tests)  
python test/test_async_robot_commands.py

# Test modular tracking system (6 tests)
python test/test_modular_tracking.py

# Test enhanced features (4 tests)
python test/test_enhanced_features.py

# Test overall module functionality (7 tests)
python test_module_functionality.py
```

### Test Coverage
- **Configuration validation**: All 14+ presets load correctly (including CharUco configurations)
- **Strategy switching**: Runtime algorithm changes work properly
- **Async robot control**: Non-blocking behavior validated
- **Hardware abstraction**: Graceful fallback when hardware missing
- **Calibration accuracy**: Transformation matrix calculation verification (coordinate system corrected ✅)
- **Performance benchmarks**: FPS and timing validation
- **CharUco detection**: Pattern detection and pose estimation accuracy
- **Coordinate transformations**: Unit conversion and axis alignment validation

## Development Workflow

### Adding New Detection Strategies
1. **Inherit from ClusteringStrategy**: Implement required methods
2. **Add to StrategyFactory**: Register new strategy
3. **Create configuration preset**: Add to ConfigFactory
4. **Write tests**: Validate functionality

### Adding New Configurations
1. **Define in ConfigFactory.PRESETS**: Specify parameters
2. **Test configuration**: Validate with test suite
3. **Document usage**: Add examples to applications

### Hardware Integration
1. **Sensor managers**: Inherit from SensorManager base class
2. **Robot controllers**: Inherit from RobotController base class  
3. **Configuration support**: Add hardware-specific parameters

## Performance Characteristics

### Real-time Performance (BEV Strategy)
- **Detection FPS**: 30+ fps
- **Visualization FPS**: 30 fps (maintained during robot movement)
- **Robot command latency**: <100ms to queue, non-blocking execution
- **Memory usage**: ~200MB typical

### High-Quality Performance (DBSCAN Strategy)  
- **Detection FPS**: ~5 fps
- **Point cloud processing**: Full resolution, no downsampling
- **Memory usage**: ~500MB typical

### Classification Performance (PointNet Strategy)
- **Classification FPS**: ~1 fps  
- **Model loading**: ~2 seconds initial startup
- **GPU acceleration**: Supported via PyTorch CUDA

## Hardware Requirements

### Recommended Setup
- **Camera**: Intel RealSense D435/D455 with IMU
- **Robot**: myCobot 280 robot arm (optional)
- **Computer**: Windows 10/11 or Ubuntu 18.04+, 8GB+ RAM
- **GPU**: CUDA-compatible for PointNet acceleration (optional)

### Minimal Setup  
- **Camera**: Any RealSense camera or bag files for replay
- **Robot**: System works without robot (detection only)
- **Computer**: Any system capable of running OpenCV and numpy

## Robot Hardware Specifications

The project is optimized for the **myCobot 280 for M5** collaborative robotic arm:

### Identity & Type
- **Model**: myCobot 280 for M5
- **Manufacturer**: Elephant Robotics
- **Type**: 6-axis collaborative robotic arm

### Performance Characteristics
- **Payload**: 250g maximum
- **Repeatability**: 0.5mm precision
- **Degrees of Freedom**: 6 joints
- **Arm Span**: 350mm reach
- **Effective Radius**: 280mm working envelope

### Physical Properties
- **Weight**: 850g (lightweight design)
- **Material**: Industrial Nylon construction
- **Power Requirements**: 8V-12V, 5A maximum
- **Operating Temperature**: -5°C to 45°C

### Kinematics & Workspace
- **Workspace Shape**: Partial sphere around base
- **Link Segments**:
  - Base to J2: 131.56mm
  - J2 to J3: 110.4mm  
  - J3 to J4: 96.0mm
  - J4 to End: 139.57mm

### Joint Limits (All in degrees)
- **J1-J5**: ±165° range each
- **J6**: ±179° range (full rotation capability)

### Controller System
- **Main Controller**: M5Stack Basic
- **Processor**: ESP32 Dual-core 240MHz
- **Communication**: USB Type-C, Wi-Fi, Bluetooth
- **Programming**: Compatible with Python, Arduino IDE, and myCobot API

### Integration Notes
- All robot control code is designed around these specifications
- Safety constraints implemented based on joint limits and workspace bounds
- Calibration algorithms account for the 0.5mm repeatability specification
- Async command system optimized for the ESP32 communication latency

## Development Guidelines

### Code Standards
- **Use modular architecture**: Extend existing strategies rather than creating monolithic scripts
- **Configuration-driven**: Use ConfigFactory presets rather than hardcoded parameters
- **Test coverage**: Write tests for new functionality
- **Hardware abstraction**: Support graceful fallback when hardware unavailable
- **Async-aware**: Use async robot commands to maintain UI responsiveness

### Python Environment
- **Always use**: `.venv/Scripts/python.exe` for consistency
- **Import style**: Use relative imports within module/, absolute for external
- **Error handling**: Graceful degradation when hardware unavailable
- **Logging**: Use provided logging framework for debugging

### Performance Optimization
- **Real-time applications**: Use BEV strategy configurations
- **High-quality applications**: Use DBSCAN or PointNet strategies  
- **Adaptive applications**: Use adaptive_tracking for automatic optimization
- **Memory management**: Large point clouds should use downsampling

## Migration from Legacy

### Legacy → Modular Migration
- **`track_object.py`** → **`modular_track_object.py`** (async robot control, strategy switching)
- **`calibrate_camera.py`** → **`modular_calibrate_camera.py`** (multiple presets, validation)
- **Sequential scripts** → **`unified_detection_app.py`** (integrated research platform)

### Backward Compatibility
- **Legacy scripts remain functional** for existing workflows
- **Configuration files compatible** between systems
- **Gradual migration supported** - can use both systems simultaneously

The project has successfully evolved from a research-focused PointNet classification system into a **production-ready modular object detection and robotics platform** with real-time performance, comprehensive testing, and professional software engineering practices.

## Task Master AI Instructions
**Import Task Master's development workflow commands and guidelines, treat as if import is in the main CLAUDE.md file.**
@./.taskmaster/CLAUDE.md
