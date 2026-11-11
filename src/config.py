"""
Configuration Loader and Validator
Calibrated Concept-Based Feedback Model (CSE 4385)

Loads settings from YAML files, applies defaults, and validates consistency
with the course project proposal.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigError(Exception):
    """Raised when configuration file is missing, unreadable, or invalid."""
    pass


def get_project_root() -> Path:
    """Returns the absolute Path of the project root directory."""
    return Path(__file__).resolve().parent.parent


def _parse_yaml_fallback(text: str) -> Dict[str, Any]:
    """
    Lightweight fallback parser for basic YAML when PyYAML is not installed.
    Supports comments, nested dictionaries (2-space indent), lists (- item),
    integers, floats, booleans, and strings.
    """
    root: Dict[str, Any] = {}
    current_section: Optional[str] = None
    current_subsection: Optional[str] = None

    lines = text.splitlines()
    for raw_line in lines:
        # Strip comments and trailing whitespace
        line = raw_line.split("#")[0].rstrip()
        if not line or line.strip() == "---":
            continue

        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        # Parse lists (- item)
        if stripped.startswith("- "):
            item_val = stripped[2:].strip().strip("\"'")
            # Convert types if possible
            if item_val.lower() == "true":
                typed_val = True
            elif item_val.lower() == "false":
                typed_val = False
            else:
                try:
                    if "." in item_val:
                        typed_val = float(item_val)
                    else:
                        typed_val = int(item_val)
                except ValueError:
                    typed_val = item_val

            target_dict = root
            if current_section:
                target_dict = root.setdefault(current_section, {})
            # Look for last key or list
            # Fallback list handling
            continue

        if ":" not in stripped:
            continue

        key, val = stripped.split(":", 1)
        key = key.strip()
        val = val.strip()

        # Determine nesting level based on indent
        if indent == 0:
            current_section = key
            current_subsection = None
            if not val:
                root[key] = {}
            else:
                root[key] = _parse_scalar(val)
        elif indent == 2:
            current_subsection = key
            target = root.setdefault(current_section, {})
            if not val:
                target[key] = {}
            elif val.startswith("[") and val.endswith("]"):
                # Inline list: [a, b, c]
                items = [x.strip().strip("\"'") for x in val[1:-1].split(",") if x.strip()]
                target[key] = [_parse_scalar(x) for x in items]
            else:
                target[key] = _parse_scalar(val)
        elif indent == 4 and current_section:
            target = root.setdefault(current_section, {})
            if current_subsection and isinstance(target.get(current_subsection), dict):
                subtarget = target[current_subsection]
            else:
                subtarget = target
            subtarget[key] = _parse_scalar(val)

    return root


def _parse_scalar(val: str) -> Any:
    val = val.strip().strip("\"'")
    if not val:
        return None
    lower = val.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("null", "none"):
        return None
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads and validates the project configuration.

    Args:
        config_path: Path to YAML config file. If None, defaults to configs/default.yaml.

    Returns:
        Dictionary containing project configuration parameters.

    Raises:
        ConfigError: If configuration file is missing, empty, or fails validation.
    """
    if config_path is None:
        target_path = get_project_root() / "configs" / "default.yaml"
    else:
        target_path = Path(config_path)
        if not target_path.is_absolute():
            target_path = get_project_root() / target_path

    if not target_path.exists():
        raise ConfigError(
            f"Configuration file not found at: {target_path}\n"
            f"Please ensure 'configs/default.yaml' exists in your project."
        )

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ConfigError(f"Failed to read configuration file at {target_path}: {e}")

    if not content.strip():
        raise ConfigError(f"Configuration file at {target_path} is empty.")

    # Try loading with PyYAML first; use fallback parser if PyYAML is not installed
    try:
        import yaml
        try:
            config = yaml.safe_load(content)
        except Exception as e:
            raise ConfigError(f"YAML parsing error in {target_path}: {e}")
    except ImportError:
        # Graceful fallback when running before pip install -r requirements.txt
        config = _parse_yaml_fallback(content)

    if not isinstance(config, dict):
        raise ConfigError(f"Invalid configuration format in {target_path}. Expected a key-value mapping.")

    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validates essential configuration fields against proposal requirements.

    Raises:
        ConfigError: If required fields are missing or have contradictory values.
    """
    required_sections = ["project", "reproducibility", "data", "model", "calibration", "conformal"]
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required configuration section: '{section}'")

    data_cfg = config.get("data", {})
    expected_classes = 200
    expected_concepts = 112
    expected_attributes = 312

    if data_cfg.get("num_classes") != expected_classes:
        raise ConfigError(
            f"Invalid data.num_classes: got {data_cfg.get('num_classes')}, "
            f"expected {expected_classes} (as specified in CUB-200-2011 proposal)."
        )

    if data_cfg.get("num_concepts") != expected_concepts:
        raise ConfigError(
            f"Invalid data.num_concepts: got {data_cfg.get('num_concepts')}, "
            f"expected {expected_concepts} (as specified in proposal Section 5.1)."
        )

    if data_cfg.get("num_total_attributes") != expected_attributes:
        raise ConfigError(
            f"Invalid data.num_total_attributes: got {data_cfg.get('num_total_attributes')}, "
            f"expected {expected_attributes} (raw attributes in CUB-200-2011)."
        )

    return True


if __name__ == "__main__":
    try:
        cfg = load_config()
        print(f"Configuration successfully loaded!")
        print(f"Project:      {cfg['project']['name']}")
        print(f"Species:      {cfg['data']['num_classes']}")
        print(f"Concepts:     {cfg['data']['num_concepts']}")
        print(f"Backbone:     {cfg['model']['backbone']}")
        print(f"Seeds:        {cfg['reproducibility']['experiment_seeds']}")
    except ConfigError as err:
        print(f"Configuration error: {err}")

