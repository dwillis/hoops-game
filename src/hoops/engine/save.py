"""Save/load game state to disk."""
from __future__ import annotations

import json
from pathlib import Path


_SAVES_DIR = Path.home() / ".hoops" / "saves"


def saves_dir() -> Path:
    """Return the saves directory path."""
    return _SAVES_DIR


def save_path_for(home_name: str, away_name: str, season: str) -> Path:
    """Return the save file path for a given matchup."""
    safe_home = home_name.replace(" ", "_")
    safe_away = away_name.replace(" ", "_")
    return _SAVES_DIR / f"{safe_home}_vs_{safe_away}_{season}.json"


def has_save(home_name: str, away_name: str, season: str) -> bool:
    """Check if a save file exists for the given matchup."""
    return save_path_for(home_name, away_name, season).exists()


def save_game(data: dict, home_name: str, away_name: str, season: str) -> Path:
    """Write a save dict to disk. Returns the path written."""
    path = save_path_for(home_name, away_name, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path


def load_save(path: Path) -> dict:
    """Read a save dict from disk."""
    return json.loads(path.read_text())
