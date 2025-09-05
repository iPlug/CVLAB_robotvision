# LLM Grabber - Intelligent Robot Control System

An isolated robot control system designed for LLM function calling integration, extracted from the main CVLAB robotvision project.

## Overview

This system provides a clean interface for robot control with computer vision capabilities, specifically designed to be extended with LLM function calling for natural language robot commands.

## Features

- Real-time object detection and tracking
- Advanced robot motion control with state machine
- Grasping and manipulation capabilities
- Coordinate transformations between camera and robot space
- Non-blocking robot operations
- Safety constraints and validation

## Setup

1. **Environment**: Use Python 3.8.0 with virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate.bat  # Windows
```

2. **Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Hardware Requirements**:
   - Intel RealSense camera (D435/D455 recommended)
   - myCobot 280 robot arm (optional, system works without robot)

## Usage

### Basic Operation
```bash
python main.py
```

### Keyboard Controls
- `g`: Start grasping sequence for closest object
- `1-9`: Grasp specific object by ID
- `h`: Move robot to home position
- `s`: Stop current robot movement
- `q`: Quit application

## Architecture

```
llm_grabber/
├── main.py                    # Main application entry point
├── robot_motions.py          # Robot motion controller (extracted functions)
├── utils_modules/            # Utility modules for vision and robot control
├── module/robot/            # Robot controller implementations
└── requirements.txt         # Dependencies
```

## Robot Motion Functions

The `robot_motions.py` module provides these key functions for LLM integration:

- `grasp_object(object_id=None)` - Grasp target object
- `move_to_position(x, y, z)` - Move to specific coordinates  
- `open_gripper()` / `close_gripper()` - Gripper control
- `move_to_home()` - Return to home position
- `stop_current_action()` - Emergency stop
- `get_robot_status()` - Get current robot state

## LLM Integration (Planned)

This system is designed to be extended with:
- Natural language command processing
- Function calling interface for LLMs
- Voice command integration
- Safety validation for AI-generated commands

## Coordinate Systems

- **Camera**: RealSense coordinate system (meters)
- **Robot**: myCobot base coordinates (millimeters)
- **Transformations**: Automatic conversion between coordinate spaces

## Safety Features

- Joint limit validation
- Workspace boundary checking
- Emergency stop functionality
- Collision avoidance (height-based)

## Development

To extend robot motions:
1. Add new functions to `robot_motions.py`
2. Follow existing patterns for async/sync operations
3. Include proper error handling and status reporting
4. Add docstrings for LLM function metadata

## Based On

Extracted and refactored from the main CVLAB robotvision project's `lidar_realtime_prediction_with_robot.py` system.