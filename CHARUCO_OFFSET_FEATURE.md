# CharUco Transformer - Prediction Offset Feature

## Overview

The CharUco Transformer now includes a **prediction offset** feature that allows you to add a configurable 3D offset to all position predictions. This is useful for:

- **Fine-tuning predictions** without recalibrating
- **Accounting for mounting offsets** or mechanical tolerances  
- **Correcting systematic errors** in specific setups
- **Adjusting for different gripper tools** or end-effector configurations

## Key Features

✅ **Default Offset**: `[0, 0, 0]` mm (no modification)  
✅ **Constructor Support**: Set offset during transformer creation  
✅ **Dynamic Control**: Change offset at runtime  
✅ **Validation**: Ensures proper `[x, y, z]` format  
✅ **Coordinate System**: Robot coordinates (X=forward, Y=left, Z=up)  
✅ **Units**: Millimeters (same as prediction output)  
✅ **Integration**: Works with all convenience functions  

## Usage Examples

### 1. Set Offset During Creation

```python
from charuco_transformer import CharUcoTransformer

# Create transformer with 5mm forward, 2mm left offset
transformer = CharUcoTransformer(
    calibration_path='calibration_dir',
    prediction_offset=[5.0, 2.0, 0.0]
)

# All predictions will have this offset applied
position = transformer.predict_robot_position(image, robot_pose)
# position now includes the [5, 2, 0] mm offset
```

### 2. Set Offset Dynamically

```python
# Create transformer without offset
transformer = CharUcoTransformer('calibration_dir')

# Set offset later
transformer.set_prediction_offset([3.0, -1.5, 0.5])

# Check current offset
current_offset = transformer.get_prediction_offset()
print(f"Current offset: {current_offset}")  # [3.0, -1.5, 0.5]

# Clear offset (set to [0, 0, 0])
transformer.clear_prediction_offset()
```

### 3. Convenience Functions with Offset

```python
from charuco_transformer import create_prediction_transformer, quick_charuco_prediction

# Create transformer with offset
transformer = create_prediction_transformer(
    calibration_path='calibration_dir',
    prediction_offset=[2.0, 1.0, -0.5]
)

# Quick prediction with offset
position = quick_charuco_prediction(
    image, robot_pose, 'calibration_dir',
    prediction_offset=[5.0, 0.0, 2.0]
)
```

## API Reference

### Constructor Parameter

```python
CharUcoTransformer(
    calibration_path=None,
    prediction_offset=[x, y, z]  # 3D offset in mm
)
```

### Methods

#### `set_prediction_offset(offset)`
Set the prediction offset that will be added to all position predictions.

**Parameters:**
- `offset`: List or array `[x, y, z]` in millimeters

**Example:**
```python
# Add 5mm forward, 2mm left, 1mm up offset
transformer.set_prediction_offset([5.0, 2.0, 1.0])
```

#### `get_prediction_offset()`
Get the current prediction offset.

**Returns:** 
- `numpy.ndarray`: Current 3D offset `[x, y, z]` in mm

**Example:**
```python
offset = transformer.get_prediction_offset()
print(f"Current offset: [{offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}] mm")
```

#### `clear_prediction_offset()`
Clear the prediction offset (set to `[0, 0, 0]`).

**Example:**
```python
transformer.clear_prediction_offset()
```

## Coordinate System

The offset uses the **robot coordinate system**:

- **X-axis**: Forward direction (positive = forward from robot base)
- **Y-axis**: Side direction (positive = left when facing forward)  
- **Z-axis**: Vertical direction (positive = upward from base)
- **Units**: Millimeters

## Practical Use Cases

### 1. Correcting Systematic Error

If you notice your robot consistently grabs 3mm too far forward:

```python
# Correct by moving predictions 3mm backward
transformer.set_prediction_offset([-3.0, 0.0, 0.0])
```

### 2. Different Gripper Tools

When switching between different end-effectors:

```python
# Small gripper: no offset needed
transformer.set_prediction_offset([0.0, 0.0, 0.0])

# Large gripper: account for size difference
transformer.set_prediction_offset([10.0, 5.0, -2.0])
```

### 3. Mounting Offset Compensation

For camera mounting tolerances:

```python
# Camera mounted 2mm off-center
transformer.set_prediction_offset([0.0, 2.0, 0.0])
```

### 4. Application-Specific Adjustments

For pick-and-place optimization:

```python
# Approach point slightly above object
transformer.set_prediction_offset([0.0, 0.0, 5.0])  # 5mm higher
```

## Integration with Applications

The offset feature is automatically integrated into existing applications:

### CharUco Real-time Prediction
```python
# In charuco_realtime_prediction.py applications
predictor = RealtimeCharucoPredictor()
predictor.charuco_transformer.set_prediction_offset([2.0, -1.0, 0.0])
```

### Custom Applications
```python
# Direct usage in custom scripts
transformer = CharUcoTransformer('calib_dir', prediction_offset=[5, 0, 0])
while True:
    position = transformer.predict_robot_position(image, robot_pose)
    # position includes the offset automatically
```

## Calibration Info Integration

The offset is included in calibration information:

```python
info = transformer.get_calibration_info()
print(f"Prediction offset: {info['prediction_offset_mm']} mm")
```

## Validation and Error Handling

The offset feature includes robust validation:

```python
# Valid offsets
transformer.set_prediction_offset([1.0, 2.0, 3.0])        # List
transformer.set_prediction_offset(np.array([1, 2, 3]))    # Array

# Invalid offsets (will raise ValueError)
transformer.set_prediction_offset([1, 2])           # Wrong shape
transformer.set_prediction_offset([[1, 2, 3]])      # Wrong dimensions
transformer.set_prediction_offset("invalid")        # Wrong type
```

## Performance Impact

✅ **Minimal Overhead**: Simple vector addition (O(1))  
✅ **No Detection Impact**: Applied only after successful detection  
✅ **Memory Efficient**: Stores only 3 float values  
✅ **Thread Safe**: No shared state modifications during prediction  

## Best Practices

1. **Start with [0, 0, 0]**: Use no offset initially, add only if needed
2. **Small Increments**: Adjust in small steps (1-5mm) for fine-tuning
3. **Document Offsets**: Keep track of offset values for different setups
4. **Test Thoroughly**: Validate offset effects with actual robot movements
5. **Coordinate Awareness**: Remember robot coordinate system orientation

## Migration Guide

Existing code will continue to work without changes (default offset is `[0, 0, 0]`).

### Before (no offset support):
```python
transformer = CharUcoTransformer('calibration_dir')
position = transformer.predict_robot_position(image, robot_pose)
# Manual correction in application code
corrected_position = position + np.array([5.0, 2.0, 0.0])
```

### After (with offset support):
```python
transformer = CharUcoTransformer('calibration_dir', prediction_offset=[5.0, 2.0, 0.0])
position = transformer.predict_robot_position(image, robot_pose)
# Offset applied automatically, no manual correction needed
```

---

**The prediction offset feature provides flexible, easy-to-use fine-tuning capabilities while maintaining full backward compatibility with existing code.**