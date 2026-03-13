# Filename: con.py
# Author: AK Wash
# Created: 2026-03-10

# Description: function for loading the configuration files, 
# the configuration files are stored in yaml. Settings for:

# - dataset structure
# - preprocessing parameters
# - model architecture
# - training hyperparameters

# used by: prepreocessing, dataset loaders, model training

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)