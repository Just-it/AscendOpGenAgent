"""Shared config loader for framework reference tables.

`hallucinated_scripts` (alias map for fat-fingered script names) lives in
`.autoresearch/config.yaml`. This module reads it once per process and
exposes a typed accessor.
"""
from functools import lru_cache
import os
from typing import Dict

import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTORESEARCH_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_CONFIG_PATH = os.path.join(_AUTORESEARCH_DIR, "config.yaml")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    return data


@lru_cache(maxsize=1)
def _raw() -> dict:
    """Load config.yaml once. Missing file is a hard error — the framework
    ships with one and several modules depend on it."""
    return _load_yaml(_CONFIG_PATH)


def hallucinated_scripts() -> Dict[str, str]:
    return dict(_raw().get("hallucinated_scripts", {}))
