"""Configuration utilities for the adaptive mapping framework."""
import yaml
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, output_path):
    """Save configuration to YAML file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
