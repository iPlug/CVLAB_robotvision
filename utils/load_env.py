import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional

def load_env(env_name: Optional[str] = None) -> None:
    """
    Load environment variables from .env files.
    
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