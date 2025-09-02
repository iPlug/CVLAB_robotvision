# CharUco-LiDAR Integration System

The new **CharUco-LiDAR Integration System** has been organized into its own dedicated folder:

📁 **`charuco-lidar-integration/`**

## Quick Access

### Core Files
- **`charuco-lidar-integration/charuco_lidar_bridge.py`** - Bridge training system
- **`charuco-lidar-integration/unified_position_predictor.py`** - Markerless detection
- **`charuco-lidar-integration/charuco_lidar_validator.py`** - Accuracy validation
- **`charuco-lidar-integration/CHARUCO_LIDAR_INTEGRATION.md`** - Complete documentation

## Quick Start Commands

### Training (One-time setup)
```bash
cd charuco-lidar-integration
python charuco_lidar_bridge.py --train
```

### Validation (Accuracy testing)
```bash
cd charuco-lidar-integration  
python charuco_lidar_validator.py --bridge-config ../bridge_config.json
```

### Production (Markerless detection)
```bash
cd charuco-lidar-integration
python unified_position_predictor.py --bridge-config ../bridge_config.json --robot-control
```

## Calibration Matrix

The system now uses the **latest CharUco calibration matrix**:
- **Path**: `../charuco-pose-estimation/calibration_result_1755703572/eye_in_hand_transform.npy`
- **Quality**: Fair (21 calibration points)
- **Coordinate System**: Center origin (required for bridge training)
- **Camera Intrinsics**: Custom intrinsics included in calibration folder

## What This System Does

✅ **Markerless object detection** with CharUco-level accuracy  
✅ **Real-time position prediction** without visible markers  
✅ **Robot integration** using existing calibration matrices  
✅ **Comprehensive validation** and quality assessment  

## Key Innovation

Places object on **CENTER SQUARE** of CharUco pattern during training to create spatial bridge between optical (CharUco) and LiDAR coordinate systems. After training, removes pattern and achieves accurate markerless detection.

---

📖 **For complete documentation**: See `charuco-lidar-integration/CHARUCO_LIDAR_INTEGRATION.md`