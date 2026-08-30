"""App that reads settings from a JSON file that does not exist yet."""

import json
from pathlib import Path

SETTINGS_PATH = Path("settings.json")


def load_settings() -> dict:
    """Load settings.json; the file must define the deployed defaults."""
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
