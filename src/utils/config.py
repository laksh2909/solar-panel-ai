"""
Configuration Loader Module
Provides utilities for reading and resolving project configurations from YAML.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Returns the root directory of the solar-panel-ai project."""
    # Assuming this file is located at src/utils/config.py
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """
    Loads and parses the YAML configuration file.
    
    Args:
        config_path: Optional path to config file. Defaults to configs/config.yaml.
        
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    root = get_project_root()
    if config_path is None:
        cand1 = root / "src" / "utils" / "config.yaml"
        cand2 = root / "configs" / "config.yaml"
        config_path = cand1 if cand1.exists() else cand2
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
