import os
from dotenv import load_dotenv
from pathlib import Path

def load_env(env_name: str = None):
    """
    Load env depending on env_name.
    First load the default env then overwrite it with the provided env exist
    env_name: str = env name
    """
    base_dir = Path(__file__).resolve().parent.parent

    # Load default env if exist
    default_env_path = base_dir / ".env_default"
    if default_env_path.exists():
        load_dotenv(dotenv_path=default_env_path, override=False)
        print(f'The default env have been load')
    else:
        raise FileNotFoundError(f'The default env file don\'t exists. Make sure the file {default_env_path} exists.')
    
    # Load supplementary env if exist
    if env_name:
        other_env_path = base_dir / f'.env_{env_name}'
        if other_env_path.exists():
            load_dotenv(dotenv_path=other_env_path, override=True)
            print(f'The env file {other_env_path} has override the default env')
        else:
            print(f'The env file {other_env_path} don\'t exist, and didn\'t load')