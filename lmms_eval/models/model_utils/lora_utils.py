"""Utilities for detecting and loading LoRA adapter checkpoints."""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger as eval_logger


def is_lora_checkpoint(path: str) -> bool:
    """
    Check if the given path contains a LoRA adapter checkpoint.

    Args:
        path: Path to check for LoRA adapter files

    Returns:
        True if path contains adapter_config.json and adapter weights, False otherwise
    """
    if not path or not os.path.exists(path):
        return False

    path_obj = Path(path)

    # Check for adapter_config.json
    adapter_config_path = path_obj / "adapter_config.json"
    if not adapter_config_path.exists():
        return False

    # Check for adapter weights (safetensors or bin format)
    has_safetensors = (path_obj / "adapter_model.safetensors").exists()
    has_bin = (path_obj / "adapter_model.bin").exists()

    return has_safetensors or has_bin


def get_base_model_from_adapter_config(adapter_path: str) -> Optional[str]:
    """
    Extract the base model path from adapter_config.json.

    Args:
        adapter_path: Path to the LoRA adapter checkpoint directory

    Returns:
        Base model name or path from adapter_config.json, or None if not found
    """
    adapter_config_path = Path(adapter_path) / "adapter_config.json"

    if not adapter_config_path.exists():
        eval_logger.warning(f"adapter_config.json not found at {adapter_config_path}")
        return None

    try:
        with open(adapter_config_path, encoding="utf-8") as f:
            adapter_config = json.load(f)

        base_model = adapter_config.get("base_model_name_or_path")

        if not base_model:
            eval_logger.warning("base_model_name_or_path not found in adapter_config.json")
            return None

        eval_logger.info(f"Found base model in adapter config: {base_model}")
        return base_model

    except (json.JSONDecodeError, OSError) as e:
        eval_logger.error(f"Error reading adapter_config.json: {e}")
        return None


def detect_and_resolve_lora_path(pretrained: str) -> Tuple[str, Optional[str]]:
    """
    Detect if pretrained path is a LoRA checkpoint and resolve base model path.

    Args:
        pretrained: Path or model name to check

    Returns:
        Tuple of (base_model_path, lora_adapter_path)
        - If LoRA detected: (base_model_from_config, pretrained_path)
        - If not LoRA: (pretrained, None)
    """
    if is_lora_checkpoint(pretrained):
        base_model = get_base_model_from_adapter_config(pretrained)
        if base_model:
            eval_logger.info(f"LoRA adapter detected. Loading base model '{base_model}' with adapter from '{pretrained}'")
            return base_model, pretrained
        else:
            eval_logger.warning(f"LoRA checkpoint detected at '{pretrained}' but could not extract base model. Falling back to standard loading.")

    return pretrained, None
