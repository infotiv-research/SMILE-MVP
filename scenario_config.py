"""Shared, config-driven paths for scenario extraction and replay tools."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config():
    with (ROOT / "mvp-config.json").open() as source:
        config = json.load(source)
    if "scenario_paths" not in config:
        raise KeyError("mvp-config.json is missing the scenario_paths section")
    return config


def scenario_paths():
    return load_config()["scenario_paths"]


def path(key):
    paths = scenario_paths()
    if key not in paths:
        raise KeyError("mvp-config.json scenario_paths is missing '{}'".format(key))
    return resolve_path(paths[key])
