import yaml
from pathlib import Path

def get_config_dict(path:str):
    # this reads the yaml file at path and return a python dict

    # check the path exists:
    config_path = Path(path)
    if config_path.exists():
        with open(config_path, "r") as f: # open it
            config_dict = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    return config_dict

