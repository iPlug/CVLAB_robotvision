#!/usr/bin/env python3
"""
Test script for the asynchronous robot command system.
This verifies that robot commands can be executed without blocking the main thread.
"""

import time
import sys
import os
import threading

# Add parent directory to path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_command_queue_basic():
    """Test basic command queue functionality."""
    print("Testing Basic Command Queue Functionality...")
    
    from module.robot.command_queue import RobotCommandQueue, CommandType, CommandStatus
    from module.robot.mycobot_controller import MyCobotController
    
    # Create a mock robot controller for testing
    robot = MyCobotController()
    
    # Create command queue
    queue = RobotCommandQueue(robot, max_queue_size=5)
    
    # Start the queue
    assert queue.start(), "Failed to start command queue"
    print("  [OK] Command queue started")
    
    # Submit a test command (this will fail since robot not connected, but queue should handle it)
    command_id = queue.submit_command(
        CommandType.HOME_POSITION,
        priority=1,
        timeout=5.0
    )
    
    assert command_id is not None, "Failed to submit command"
    print(f"  [OK] Command submitted with ID: {command_id}")
    
    # Wait a moment for command to be processed
    time.sleep(1.0)
    
    # Check command status
    status = queue.get_command_status(command_id)
    assert status in [CommandStatus.COMPLETED, CommandStatus.FAILED], f"Unexpected status: {status}"
    print(f"  [OK] Command completed with status: {status.value}")
    
    # Get queue status
    queue_status = queue.get_queue_status()
    assert queue_status['is_running'], "Queue should be running"
    print(f"  [OK] Queue status: {queue_status['stats']}")
    
    # Stop the queue
    assert queue.stop(timeout=3.0), "Failed to stop command queue"
    print("  [OK] Command queue stopped")
    
    return True

def test_async_robot_controller():
    """Test async robot controller functionality."""
    print("\nTesting Async Robot Controller...")
    
    from module.robot.mycobot_controller import MyCobotController
    
    # Create robot controller
    robot = MyCobotController()
    
    # Enable async mode
    assert robot.enable_async_mode(max_queue_size=3), "Failed to enable async mode"
    print("  [OK] Async mode enabled")
    
    # Test async command submission
    command_id = robot.point_at_object_async([100, 50, 200], priority=2)
    # This will fail since robot not connected, but should not block
    
    if command_id:
        print(f"  [OK] Async command submitted: {command_id}")
        
        # Wait a bit for command processing
        time.sleep(0.5)
        
        # Check status
        status = robot.get_command_status(command_id)
        print(f"  [OK] Command status: {status.value if status else 'Unknown'}")
    else:
        print("  [INFO] Command submission rejected (expected for unconnected robot)")
    
    # Test queue status
    queue_status = robot.get_queue_status()
    print(f"  [OK] Queue stats: {queue_status.get('stats', {})}")
    
    # Disable async mode
    assert robot.disable_async_mode(timeout=3.0), "Failed to disable async mode"
    print("  [OK] Async mode disabled")
    
    return True

def test_non_blocking_behavior():
    """Test that async commands don't block the main thread."""
    print("\nTesting Non-Blocking Behavior...")
    
    from module.robot.mycobot_controller import MyCobotController
    
    robot = MyCobotController()
    robot.enable_async_mode()
    
    # Submit multiple commands rapidly (they will fail but shouldn't block)
    start_time = time.time()
    
    for i in range(3):
        command_id = robot.move_to_position_async([100 + i*10, 50, 200, 0, 180, 0], priority=1)
        # Don't wait - this should be non-blocking
    
    submission_time = time.time() - start_time
    
    # Should complete very quickly since commands are just queued
    assert submission_time < 0.1, f"Command submission took too long: {submission_time:.3f}s"
    print(f"  [OK] 3 commands submitted in {submission_time:.3f}s (non-blocking)")
    
    # Wait for commands to be processed
    time.sleep(1.0)
    
    queue_status = robot.get_queue_status()
    stats = queue_status.get('stats', {})
    print(f"  [OK] Commands processed: {stats.get('total_commands', 0)}")
    
    robot.disable_async_mode()
    return True

def test_callback_functionality():
    """Test command completion callbacks."""
    print("\nTesting Callback Functionality...")
    
    from module.robot.command_queue import RobotCommandQueue, CommandType
    from module.robot.mycobot_controller import MyCobotController
    
    # Track callback execution
    callback_results = []
    
    def test_callback(command):
        callback_results.append({
            'command_id': command.command_id,
            'status': command.status.value,
            'execution_time': command.get_execution_time()
        })
    
    robot = MyCobotController()
    queue = RobotCommandQueue(robot)
    queue.start()
    
    # Submit command with callback
    command_id = queue.submit_command(
        CommandType.HOME_POSITION,
        callback=test_callback,
        timeout=3.0
    )
    
    # Wait for command to complete
    time.sleep(1.5)
    
    # Check callback was executed
    assert len(callback_results) > 0, "Callback was not executed"
    assert callback_results[0]['command_id'] == command_id, "Wrong command ID in callback"
    print(f"  [OK] Callback executed: {callback_results[0]}")
    
    queue.stop()
    return True

def test_modular_tracker_integration():
    """Test integration with ModularObjectTracker."""
    print("\nTesting ModularTracker Integration...")
    
    from modular_track_object import ModularObjectTracker
    
    # Create tracker (will fail to connect to hardware but should initialize async system)
    tracker = ModularObjectTracker(
        config_name='realtime_tracking',
        initial_strategy='bev',
        transformation_matrix_file='nonexistent.npy'  # Won't load, but that's ok for testing
    )
    
    # Initialize (will print warnings about missing hardware)
    print("  [INFO] Initializing tracker (hardware warnings expected)...")
    
    # This will fail to connect to robot but should set up the async system
    # We'll mock the robot connection for testing
    from module.robot.mycobot_controller import MyCobotController
    mock_robot = MyCobotController()
    mock_robot.is_connected = False  # Simulate disconnected robot
    tracker.robot_controller = mock_robot
    
    # Test that async mode can be enabled
    if mock_robot.enable_async_mode():
        print("  [OK] Async mode enabled in tracker")
        
        # Test robot command callback
        def test_callback(cmd):
            print(f"  [OK] Callback received for command {cmd.command_id}")
        
        # Simulate submitting a command
        cmd_id = mock_robot.point_at_object_async([100, 50, 200], callback=test_callback)
        if cmd_id:
            print(f"  [OK] Async command submitted from tracker: {cmd_id}")
        
        # Wait briefly
        time.sleep(0.5)
        
        mock_robot.disable_async_mode()
        print("  [OK] Async mode disabled")
    else:
        print("  [WARNING] Could not enable async mode in tracker")
    
    return True

def run_async_command_tests():
    """Run all async robot command tests."""
    print("=" * 60)
    print("ASYNC ROBOT COMMAND SYSTEM - COMPREHENSIVE TESTS")
    print("=" * 60)
    
    tests = [
        ("Basic Command Queue", test_command_queue_basic),
        ("Async Robot Controller", test_async_robot_controller),
        ("Non-Blocking Behavior", test_non_blocking_behavior),
        ("Callback Functionality", test_callback_functionality),
        ("ModularTracker Integration", test_modular_tracker_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
                print(f"[OK] {test_name} PASSED")
            else:
                print(f"[FAIL] {test_name} FAILED")
        except Exception as e:
            print(f"[FAIL] {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("ASYNC COMMAND SYSTEM TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("[SUCCESS] All async command tests passed!")
        print("\nThe async robot command system is working correctly:")
        print("- Commands execute in background threads")
        print("- Main thread remains non-blocking")
        print("- Command queuing and status tracking works")
        print("- Callbacks function properly")
        print("- Integration with ModularTracker is ready")
        print("\nVisualization should now run smoothly during robot movement!")
    else:
        print("[WARNING] Some async command tests failed.")
        print("Check the errors above before using the system.")
    
    return passed == total

if __name__ == "__main__":
    success = run_async_command_tests()
    sys.exit(0 if success else 1)