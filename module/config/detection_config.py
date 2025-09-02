"""
Base configuration class for object detection systems.
"""

from typing import Dict, Any, Optional
import json
import os


class DetectionConfig:
    """
    Base configuration class for object detection parameters.
    
    Provides common configuration management for:
    - Parameter storage and retrieval
    - Configuration validation
    - File I/O operations
    - Default value management
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize detection configuration.
        
        Args:
            config_dict: Optional dictionary with configuration parameters
        """
        # Base configuration parameters
        self._config = {
            # General parameters
            'enable_timing': True,
            'enable_visualization': True,
            'enable_logging': True,
            
            # Performance parameters
            'max_processing_time': 1.0,  # Maximum processing time per frame (seconds)
            'target_fps': 30.0,
            
            # Output parameters
            'output_format': 'dict',  # 'dict', 'json', 'csv'
            'save_results': False,
            'results_directory': 'results/',
            
            # Debug parameters
            'debug_mode': False,
            'verbose_logging': False
        }
        
        # Update with provided configuration
        if config_dict:
            self.update_config(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration parameter.
        
        Args:
            key: Parameter key (supports nested keys with dots)
            default: Default value if key not found
            
        Returns:
            Parameter value or default
        """
        # Support nested keys like 'clustering.min_size'
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration parameter.
        
        Args:
            key: Parameter key (supports nested keys with dots)
            value: Parameter value
        """
        # Support nested keys like 'clustering.min_size'
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def update_config(self, config_dict: Dict[str, Any]):
        """
        Update configuration with new parameters.
        
        Args:
            config_dict: Dictionary with new parameters
        """
        self._config.update(config_dict)
    
    @property
    def config(self) -> Dict[str, Any]:
        """
        Get configuration dictionary (read-only property).
        
        Returns:
            Configuration dictionary
        """
        return self._config
    
    def get_all_config(self) -> Dict[str, Any]:
        """
        Get all configuration parameters.
        
        Returns:
            Complete configuration dictionary
        """
        return self._config.copy()
    
    def validate_config(self) -> bool:
        """
        Validate configuration parameters.
        Base implementation always returns True.
        Override in subclasses for specific validation.
        
        Returns:
            True if configuration is valid
        """
        return True
    
    def save_to_file(self, file_path: str) -> bool:
        """
        Save configuration to JSON file.
        
        Args:
            file_path: Path to save configuration file
            
        Returns:
            True if save successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving configuration to {file_path}: {e}")
            return False
    
    def load_from_file(self, file_path: str) -> bool:
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            True if load successful
        """
        try:
            if not os.path.exists(file_path):
                print(f"Configuration file {file_path} does not exist")
                return False
            
            with open(file_path, 'r') as f:
                config_dict = json.load(f)
            
            self.update_config(config_dict)
            return True
            
        except Exception as e:
            print(f"Error loading configuration from {file_path}: {e}")
            return False
    
    def reset_to_defaults(self):
        """Reset configuration to default values."""
        # This should be overridden in subclasses
        pass
    
    def get_config_summary(self) -> str:
        """
        Get a human-readable summary of the configuration.
        
        Returns:
            Configuration summary string
        """
        lines = ["Configuration Summary:", "=" * 40]
        
        def format_dict(d, indent=0):
            result = []
            for key, value in d.items():
                prefix = "  " * indent
                if isinstance(value, dict):
                    result.append(f"{prefix}{key}:")
                    result.extend(format_dict(value, indent + 1))
                else:
                    result.append(f"{prefix}{key}: {value}")
            return result
        
        lines.extend(format_dict(self._config))
        return "\n".join(lines)
    
    def copy(self) -> 'DetectionConfig':
        """
        Create a copy of this configuration.
        
        Returns:
            New DetectionConfig instance with same parameters
        """
        return self.__class__(self._config.copy())
    
    def merge_with(self, other_config: 'DetectionConfig') -> 'DetectionConfig':
        """
        Merge this configuration with another.
        
        Args:
            other_config: Other configuration to merge with
            
        Returns:
            New configuration with merged parameters
        """
        merged_config = self._config.copy()
        merged_config.update(other_config.get_all_config())
        return self.__class__(merged_config)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"{self.__class__.__name__}({len(self._config)} parameters)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}({self._config})"