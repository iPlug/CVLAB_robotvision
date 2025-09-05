"""
Robot Command Queue System for Asynchronous Robot Control

This module provides a thread-safe command queue system that allows robot commands
to be executed in the background without blocking the main visualization loop.
"""

import threading
import queue
import time
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import logging


class CommandType(Enum):
    """Types of robot commands that can be queued."""
    MOVE_TO_POSITION = "move_to_position"
    MOVE_TO_ANGLES = "move_to_angles"
    HOME_POSITION = "home_position"
    STOP_MOVEMENT = "stop_movement"
    CUSTOM_COMMAND = "custom_command"
    LOOK_AT_TABLE = "look_at_table"
    LOOK_FORWARD = "look_forward"


class CommandStatus(Enum):
    """Status of queued commands."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RobotCommand:
    """Represents a single robot command with metadata."""
    
    def __init__(self, command_type: CommandType, args: List[Any] = None, 
                 kwargs: Dict[str, Any] = None, priority: int = 1,
                 timeout: float = 10.0, callback: Optional[Callable] = None):
        """
        Initialize a robot command.
        
        Args:
            command_type: Type of command to execute
            args: Positional arguments for the command
            kwargs: Keyword arguments for the command
            priority: Command priority (higher numbers = higher priority)
            timeout: Maximum execution time in seconds
            callback: Optional callback function to call when command completes
        """
        self.command_type = command_type
        self.args = args or []
        self.kwargs = kwargs or {}
        self.priority = priority
        self.timeout = timeout
        self.callback = callback
        
        self.command_id = id(self)  # Unique identifier
        self.status = CommandStatus.PENDING
        self.created_time = time.time()
        self.start_time = None
        self.end_time = None
        self.result = None
        self.error = None
    
    def __lt__(self, other):
        """Comparison for priority queue (higher priority first)."""
        return self.priority > other.priority
    
    def get_execution_time(self) -> Optional[float]:
        """Get command execution time in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def is_expired(self) -> bool:
        """Check if command has exceeded its timeout."""
        if self.start_time:
            return time.time() - self.start_time > self.timeout
        return time.time() - self.created_time > self.timeout * 2  # Extra grace period for pending


class RobotCommandQueue:
    """
    Thread-safe command queue for asynchronous robot control.
    
    Features:
    - Priority-based command execution
    - Timeout protection
    - Command status tracking
    - Thread-safe operation
    - Rate limiting support
    - Error handling and recovery
    """
    
    def __init__(self, robot_controller, max_queue_size: int = 10,
                 worker_thread_name: str = "RobotCommandWorker"):
        """
        Initialize the command queue.
        
        Args:
            robot_controller: Robot controller instance to execute commands
            max_queue_size: Maximum number of commands in queue
            worker_thread_name: Name for the worker thread
        """
        self.robot_controller = robot_controller
        self.max_queue_size = max_queue_size
        
        # Thread-safe command queue (priority queue)
        self.command_queue = queue.PriorityQueue(maxsize=max_queue_size)
        
        # Command history and status tracking
        self.command_history = []
        self.active_command = None
        self.history_lock = threading.Lock()
        
        # Worker thread control
        self.worker_thread = None
        self.worker_thread_name = worker_thread_name
        self.shutdown_event = threading.Event()
        self.is_running = False
        
        # Performance tracking
        self.stats = {
            'total_commands': 0,
            'successful_commands': 0,
            'failed_commands': 0,
            'cancelled_commands': 0,
            'average_execution_time': 0.0
        }
        
        # Rate limiting
        self.last_command_time = 0
        self.min_command_interval = 0.1  # Minimum time between command starts
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> bool:
        """
        Start the command queue worker thread.
        
        Returns:
            True if started successfully
        """
        if self.is_running:
            self.logger.warning("Command queue already running")
            return True
        
        try:
            self.shutdown_event.clear()
            self.worker_thread = threading.Thread(
                target=self._worker_loop,
                name=self.worker_thread_name,
                daemon=True
            )
            self.worker_thread.start()
            self.is_running = True
            self.logger.info(f"Robot command queue started with thread: {self.worker_thread_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start command queue: {e}")
            return False
    
    def stop(self, timeout: float = 5.0) -> bool:
        """
        Stop the command queue and wait for worker thread to finish.
        
        Args:
            timeout: Maximum time to wait for thread to stop
            
        Returns:
            True if stopped successfully
        """
        if not self.is_running:
            return True
        
        self.logger.info("Stopping robot command queue...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel pending commands
        self._cancel_all_pending_commands()
        
        # Wait for worker thread to finish
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=timeout)
            
            if self.worker_thread.is_alive():
                self.logger.warning("Worker thread did not stop within timeout")
                return False
        
        self.is_running = False
        self.logger.info("Robot command queue stopped")
        return True
    
    def submit_command(self, command_type: CommandType, *args, 
                      priority: int = 1, timeout: float = 10.0,
                      callback: Optional[Callable] = None, **kwargs) -> Optional[int]:
        """
        Submit a command to the queue.
        
        Args:
            command_type: Type of command to execute
            *args: Positional arguments for the command
            priority: Command priority (higher = more important)
            timeout: Command timeout in seconds
            callback: Optional completion callback
            **kwargs: Keyword arguments for the command
            
        Returns:
            Command ID if submitted successfully, None otherwise
        """
        if not self.is_running:
            self.logger.error("Cannot submit command: queue not running")
            return None
        
        # Check queue capacity
        if self.command_queue.qsize() >= self.max_queue_size:
            self.logger.warning("Command queue full, rejecting command")
            return None
        
        # Create command
        command = RobotCommand(
            command_type=command_type,
            args=list(args),
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            callback=callback
        )
        
        try:
            # Add to priority queue
            self.command_queue.put_nowait(command)
            
            # Track in history
            with self.history_lock:
                self.command_history.append(command)
                # Keep history limited
                if len(self.command_history) > 100:
                    self.command_history.pop(0)
            
            self.stats['total_commands'] += 1
            self.logger.debug(f"Command {command.command_id} submitted: {command_type.value}")
            return command.command_id
            
        except queue.Full:
            self.logger.warning("Failed to submit command: queue full")
            return None
        except Exception as e:
            self.logger.error(f"Failed to submit command: {e}")
            return None
    
    def get_command_status(self, command_id: int) -> Optional[CommandStatus]:
        """Get the status of a specific command."""
        with self.history_lock:
            for command in self.command_history:
                if command.command_id == command_id:
                    return command.status
        return None
    
    def cancel_command(self, command_id: int) -> bool:
        """Cancel a pending command (cannot cancel in-progress commands)."""
        with self.history_lock:
            for command in self.command_history:
                if command.command_id == command_id:
                    if command.status == CommandStatus.PENDING:
                        command.status = CommandStatus.CANCELLED
                        self.stats['cancelled_commands'] += 1
                        return True
        return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics."""
        with self.history_lock:
            active_cmd = self.active_command
            
        return {
            'is_running': self.is_running,
            'queue_size': self.command_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'active_command': {
                'command_id': active_cmd.command_id if active_cmd else None,
                'type': active_cmd.command_type.value if active_cmd else None,
                'status': active_cmd.status.value if active_cmd else None
            } if active_cmd else None,
            'stats': self.stats.copy(),
            'recent_commands': len(self.command_history)
        }
    
    def _worker_loop(self):
        """Main worker thread loop for processing commands."""
        self.logger.info("Robot command worker thread started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get next command with timeout
                try:
                    command = self.command_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Check if command expired while waiting
                if command.is_expired():
                    command.status = CommandStatus.FAILED
                    command.error = "Command expired before execution"
                    self.stats['failed_commands'] += 1
                    self.command_queue.task_done()
                    continue
                
                # Check if command was cancelled
                if command.status == CommandStatus.CANCELLED:
                    self.command_queue.task_done()
                    continue
                
                # Execute command
                self._execute_command(command)
                self.command_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                continue
        
        self.logger.info("Robot command worker thread stopped")
    
    def _execute_command(self, command: RobotCommand):
        """Execute a single command."""
        # Apply rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_command_time
        if time_since_last < self.min_command_interval:
            time.sleep(self.min_command_interval - time_since_last)
        
        # Update command status
        command.status = CommandStatus.IN_PROGRESS
        command.start_time = time.time()
        self.active_command = command
        self.last_command_time = command.start_time
        
        try:
            # Execute the appropriate robot controller method
            if command.command_type == CommandType.MOVE_TO_POSITION:
                result = self.robot_controller.move_to_position(*command.args, **command.kwargs)
            elif command.command_type == CommandType.MOVE_TO_ANGLES:
                result = self.robot_controller.move_to_angles(*command.args, **command.kwargs)
            elif command.command_type == CommandType.HOME_POSITION:
                result = self.robot_controller.home_position(*command.args, **command.kwargs)
            elif command.command_type == CommandType.LOOK_AT_TABLE:
                result = self.robot_controller.look_at_table(*command.args, **command.kwargs)
            elif command.command_type == CommandType.LOOK_FORWARD:
                result = self.robot_controller.look_forward(*command.args, **command.kwargs)
            elif command.command_type == CommandType.STOP_MOVEMENT:
                result = self.robot_controller.stop_movement(*command.args, **command.kwargs)
            elif command.command_type == CommandType.CUSTOM_COMMAND:
                # For custom commands, first arg should be the method name
                method_name = command.args[0]
                method = getattr(self.robot_controller, method_name)
                result = method(*command.args[1:], **command.kwargs)
            else:
                raise ValueError(f"Unknown command type: {command.command_type}")
            
            # Command completed successfully
            command.status = CommandStatus.COMPLETED
            command.result = result
            self.stats['successful_commands'] += 1
            
        except Exception as e:
            # Command failed
            command.status = CommandStatus.FAILED
            command.error = str(e)
            command.result = False
            self.stats['failed_commands'] += 1
            self.logger.error(f"Command {command.command_id} failed: {e}")
        
        finally:
            # Update timing and status
            command.end_time = time.time()
            self.active_command = None
            
            # Update average execution time
            exec_time = command.get_execution_time()
            if exec_time:
                current_avg = self.stats['average_execution_time']
                total_successful = self.stats['successful_commands']
                if total_successful > 1:
                    self.stats['average_execution_time'] = (
                        (current_avg * (total_successful - 1) + exec_time) / total_successful
                    )
                else:
                    self.stats['average_execution_time'] = exec_time
            
            # Call completion callback if provided
            if command.callback:
                try:
                    command.callback(command)
                except Exception as e:
                    self.logger.error(f"Error in command callback: {e}")
    
    def _cancel_all_pending_commands(self):
        """Cancel all pending commands during shutdown."""
        with self.history_lock:
            cancelled_count = 0
            for command in self.command_history:
                if command.status == CommandStatus.PENDING:
                    command.status = CommandStatus.CANCELLED
                    cancelled_count += 1
            
            if cancelled_count > 0:
                self.stats['cancelled_commands'] += cancelled_count
                self.logger.info(f"Cancelled {cancelled_count} pending commands during shutdown")


# Convenience functions for common robot commands
def create_point_at_object_command(camera_position: List[float], 
                                 priority: int = 2) -> Dict[str, Any]:
    """Create a command to point robot at detected object."""
    return {
        'command_type': CommandType.CUSTOM_COMMAND,
        'args': ['point_at_object', camera_position],
        'priority': priority,
        'timeout': 15.0
    }


def create_home_position_command(priority: int = 1) -> Dict[str, Any]:
    """Create a command to move robot to home position."""
    return {
        'command_type': CommandType.HOME_POSITION,
        'priority': priority,
        'timeout': 10.0
    }


def create_stop_movement_command(priority: int = 3) -> Dict[str, Any]:
    """Create a high-priority command to stop robot movement."""
    return {
        'command_type': CommandType.STOP_MOVEMENT,
        'priority': priority,
        'timeout': 5.0
    }


def create_look_at_table_command(priority: int = 1) -> Dict[str, Any]:
    """Create a command to move robot to look at table position."""
    return {
        'command_type': CommandType.LOOK_AT_TABLE,
        'priority': priority,
        'timeout': 10.0
    }