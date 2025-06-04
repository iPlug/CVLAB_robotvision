"""Environment configuration management."""
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Optional

# Required environment variables and their types
REQUIRED_ENV_VARS = {
    'BAG_FILE_ROOT_FOLDER': str,    # Root folder containing raw bag files
    'DEST_COPY_ROOT_FOLDER': str,    # Destination for processed files
    'DATASET_ROOT_FOLDER': str,      # Root folder for the final dataset
}

def validate_env() -> None:
    """
    Validate that all required environment variables are present and of correct type.
    
    Raises:
        ValueError: If any required variables are missing or of invalid type
    """
    missing_vars = []
    for var, var_type in REQUIRED_ENV_VARS.items():
        value = os.getenv(var)
        if value is None:
            missing_vars.append(var)
            continue
        try:
            var_type(value)
        except ValueError:
            missing_vars.append(f"{var} (invalid type)")
    
    if missing_vars:
        raise ValueError(
            f"Missing or invalid required environment variables: {', '.join(missing_vars)}"
        )

def get_env(var_name: str, default: Any = None) -> Any:
    """
    Get environment variable with type conversion.
    
    Args:
        var_name: Name of the environment variable
        default: Default value if variable is not set
    
    Returns:
        The environment variable value with proper type conversion
    """
    if var_name in REQUIRED_ENV_VARS:
        value = os.getenv(var_name)
        var_type = REQUIRED_ENV_VARS[var_name]
        return var_type(value) if value is not None else default
    return os.getenv(var_name, default)

def load_env(env_name: Optional[str] = None) -> None:
    """
    Load and validate environment variables from .env files.
    
    Args:
        env_name: Optional name of specific environment to load
                 (will be loaded after default)
    
    First loads the default env then optionally overwrites with specific env.
    Validates required variables are present.
    
    Raises:
        FileNotFoundError: If default env file is missing
        ValueError: If required variables are missing or invalid
    """
    base_dir = Path(__file__).resolve().parent.parent

    # Load default env if exists
    default_env_path = base_dir / ".env_default"
    if default_env_path.exists():
        load_dotenv(dotenv_path=default_env_path, override=False)
        print(f'Default environment loaded from {default_env_path}')
    else:
        raise FileNotFoundError(
            f'Default environment file not found at {default_env_path}'
        )
    
    # Load environment-specific configuration if it exists
    if env_name:
        specific_env_path = base_dir / f'.env_{env_name}'
        if specific_env_path.exists():
            load_dotenv(dotenv_path=specific_env_path, override=True)
            print(f'Environment-specific configuration loaded from {specific_env_path}')
        else:
            print(f'Environment-specific configuration not found at {specific_env_path}')
    
    # Validate environment variables
    validate_env()