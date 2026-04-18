"""
Utility to load the central YAML config.
Usage:
    from src.utils.config_loader import load_config
    cfg = load_config()
    raw_dir = cfg["data"]["raw_dir"]
"""

import yaml
from pathlib import Path
from loguru import logger

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(config_path: str | Path = _CONFIG_PATH) -> dict:
    """Load and return the YAML config as a dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.debug(f"Config loaded from {path}")
    return cfg
