#!/usr/bin/env python3
"""
LLM Integration Example for Robot Motion Controller

This demonstrates how the extracted robot motion functions can be 
integrated with LLM function calling for natural language robot control.

This is a simplified example showing the integration pattern.
In a real implementation, you would connect this to your preferred 
LLM API (OpenAI, Claude, etc.) with proper function calling setup.
"""

import json
from typing import Dict, List, Any
from robot_motions import RobotMotionController, get_available_robot_functions

class SimpleLLMRobotInterface:
    """
    Simple interface demonstrating how LLM function calling 
    can be integrated with the robot motion controller.
    """
    
    def __init__(self, robot_motion_controller: RobotMotionController):
        self.robot_controller = robot_motion_controller
        self.available_functions = get_available_robot_functions()
        
    def process_natural_language_command(self, user_command: str) -> str:
        """
        Process natural language command and convert to robot actions.
        
        In a real implementation, this would call an LLM API with function calling.
        This is a simplified demo using keyword matching.
        """
        command = user_command.lower().strip()
        
        print(f"🧠 Processing command: '{user_command}'")
        
        # Simple keyword-based command processing (demo only)
        if any(word in command for word in ['grasp', 'grab', 'pick up', 'take']):
            return self._handle_grasp_command(command)
            
        elif any(word in command for word in ['home', 'return', 'safe']):
            return self._handle_home_command()
            
        elif any(word in command for word in ['stop', 'halt', 'emergency']):
            return self._handle_stop_command()
            
        elif any(word in command for word in ['open', 'release']):
            return self._handle_open_gripper_command()
            
        elif any(word in command for word in ['close', 'grip', 'clamp']):
            return self._handle_close_gripper_command()
            
        elif any(word in command for word in ['move to', 'go to', 'position']):
            return self._handle_position_command(command)
            
        elif any(word in command for word in ['status', 'state', 'where']):
            return self._handle_status_command()
            
        else:
            return self._handle_unknown_command(command)
    
    def _handle_grasp_command(self, command: str) -> str:
        """Handle grasping-related commands."""
        # In real LLM implementation, this would be a function call
        result = self.robot_controller.grasp_object()
        
        if result:
            return "✅ Starting grasping sequence for closest object..."
        else:
            return "❌ Cannot start grasping - robot not available or already busy"
    
    def _handle_home_command(self) -> str:
        """Handle home position commands."""
        result = self.robot_controller.move_to_home()
        
        if result:
            return "✅ Moving robot to home position..."
        else:
            return "❌ Cannot move to home - robot not available"
    
    def _handle_stop_command(self) -> str:
        """Handle emergency stop commands."""
        result = self.robot_controller.stop_current_action()
        
        if result:
            return "🛑 Emergency stop activated - all movements halted"
        else:
            return "❌ Stop command failed"
    
    def _handle_open_gripper_command(self) -> str:
        """Handle gripper opening commands."""
        result = self.robot_controller.open_gripper()
        
        if result:
            return "✅ Opening gripper..."
        else:
            return "❌ Cannot open gripper - robot not available"
    
    def _handle_close_gripper_command(self) -> str:
        """Handle gripper closing commands."""
        result = self.robot_controller.close_gripper()
        
        if result:
            return "✅ Closing gripper..."
        else:
            return "❌ Cannot close gripper - robot not available"
    
    def _handle_position_command(self, command: str) -> str:
        """Handle position movement commands."""
        # In real implementation, LLM would extract coordinates from natural language
        # For demo, we'll use default coordinates
        result = self.robot_controller.move_to_position(150, 0, 100)
        
        if result:
            return "✅ Moving to position (150, 0, 100) mm..."
        else:
            return "❌ Cannot move to position - robot not available"
    
    def _handle_status_command(self) -> str:
        """Handle status query commands."""
        status = self.robot_controller.get_robot_status()
        
        status_lines = ["🤖 Robot Status:"]
        for key, value in status.items():
            if key != 'timestamp':
                status_lines.append(f"  • {key}: {value}")
        
        return "\\n".join(status_lines)
    
    def _handle_unknown_command(self, command: str) -> str:
        """Handle unrecognized commands."""
        return f"❓ I don't understand the command: '{command}'\\n\\nAvailable actions:\\n" + self._get_help_text()
    
    def _get_help_text(self) -> str:
        """Get help text showing available commands."""
        help_lines = []
        for func_name, func_info in self.available_functions.items():
            help_lines.append(f"  • {func_info['description']}")
        
        return "\\n".join(help_lines)


def demo_llm_integration():
    """Demonstrate LLM integration with robot motion controller."""
    
    print("🚀 LLM Robot Integration Demo")
    print("="*60)
    
    # Initialize robot motion controller (without actual robot hardware)
    robot_controller = RobotMotionController(
        enhanced_robot_controller=None,  # No actual robot for demo
        movement_speed=50
    )
    
    # Initialize LLM interface
    llm_interface = SimpleLLMRobotInterface(robot_controller)
    
    # Demo commands
    test_commands = [
        "Please grab the nearest object",
        "Move the robot to home position", 
        "Open the gripper",
        "Close the gripper",
        "Stop all movements immediately",
        "Move to position 100, 50, 150",
        "What's the robot status?",
        "Do a backflip"  # Unknown command
    ]
    
    print("\\n📝 Testing natural language commands:")
    print("-" * 40)
    
    for command in test_commands:
        print(f"\\n👤 User: {command}")
        response = llm_interface.process_natural_language_command(command)
        print(f"🤖 Robot: {response}")
    
    print("\\n" + "="*60)
    print("✨ Demo completed!")
    print("\\n📋 Next Steps for Real LLM Integration:")
    print("  1. Connect to LLM API (OpenAI, Claude, etc.)")
    print("  2. Set up proper function calling schema")
    print("  3. Add parameter extraction from natural language")
    print("  4. Implement safety validation for LLM commands")
    print("  5. Add conversation memory and context")
    print("  6. Connect to actual robot hardware")
    print("="*60)


if __name__ == "__main__":
    demo_llm_integration()