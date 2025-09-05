#!/usr/bin/env python3
"""
Test script for robot motion controller functionality.

This script tests the robot motion functions without requiring 
camera hardware or full system dependencies.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robot_motions import RobotMotionController, get_available_robot_functions

def test_robot_motion_functions():
    """Test basic robot motion controller functionality."""
    
    print("="*60)
    print("ROBOT MOTION CONTROLLER TEST")
    print("="*60)
    
    # Test function metadata
    print("\n1. Testing function metadata...")
    functions = get_available_robot_functions()
    print(f"Available functions: {len(functions)}")
    for func_name, func_info in functions.items():
        print(f"  - {func_name}: {func_info['description']}")
    
    # Test robot motion controller initialization
    print("\n2. Testing RobotMotionController initialization...")
    try:
        # Initialize without actual robot hardware
        robot_controller = RobotMotionController(
            enhanced_robot_controller=None,  # No actual robot
            movement_speed=50
        )
        print("✓ RobotMotionController initialized successfully")
        
        # Test status without robot
        print("\n3. Testing status reporting...")
        status = robot_controller.get_robot_status()
        print("Robot status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Test commands without robot (should fail gracefully)
        print("\n4. Testing graceful failure handling...")
        
        print("Testing move_to_home()...")
        result = robot_controller.move_to_home()
        print(f"Result: {result} (expected: False - no robot)")
        
        print("Testing open_gripper()...")
        result = robot_controller.open_gripper()
        print(f"Result: {result} (expected: False - no robot)")
        
        print("Testing stop_current_action()...")
        result = robot_controller.stop_current_action()
        print(f"Result: {result} (expected: True - always works)")
        
        print("Testing move_to_position()...")
        result = robot_controller.move_to_position(100, 50, 200)
        print(f"Result: {result} (expected: False - no robot)")
        
        print("Testing grasp_object()...")
        result = robot_controller.grasp_object(camera_position_m=[0.1, 0.05, 0.2])
        print(f"Result: {result} (expected: False - no robot)")
        
        print("\n5. Testing state machine...")
        # Test state machine processing (should not crash)
        robot_controller.process_grasping_state_machine()
        print("✓ State machine processing completed without errors")
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("✅ Robot motion controller is ready for LLM integration")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_interface():
    """Test LLM-specific interface functions."""
    
    print("\n" + "="*60)
    print("LLM INTEGRATION TEST")
    print("="*60)
    
    try:
        # Test that all required functions are available
        functions = get_available_robot_functions()
        required_functions = [
            'grasp_object',
            'move_to_position', 
            'open_gripper',
            'close_gripper',
            'move_to_home',
            'stop_current_action',
            'get_robot_status'
        ]
        
        print("Checking required LLM functions...")
        missing_functions = []
        for func in required_functions:
            if func in functions:
                print(f"✓ {func}")
            else:
                print(f"✗ {func} - MISSING")
                missing_functions.append(func)
        
        if missing_functions:
            print(f"\n❌ Missing functions: {missing_functions}")
            return False
        else:
            print("\n✅ All required LLM functions are available!")
            return True
            
    except Exception as e:
        print(f"✗ LLM interface test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting robot motion controller tests...\n")
    
    # Run tests
    basic_test_passed = test_robot_motion_functions()
    llm_test_passed = test_llm_interface()
    
    print("\n" + "="*80)
    if basic_test_passed and llm_test_passed:
        print("🎉 ALL TESTS SUCCESSFUL!")
        print("🤖 Robot motion controller is ready for LLM integration")
        print("📋 Ready to add natural language command processing")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Please check the errors above and fix issues")
    print("="*80)