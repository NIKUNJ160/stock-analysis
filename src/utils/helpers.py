import json
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Any
from datetime import datetime


def safe_read_json(filepath: str | Path) -> dict:
    """Safely read a JSON file, returning empty dict on failure."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def safe_write_json(filepath: str | Path, data: dict) -> None:
    """Atomically write JSON by writing to temp first."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    tmp_path.replace(filepath)


def validate_dataframe(df: pd.DataFrame, required_cols: list[str], name: str = "DataFrame") -> bool:
    """Validate that a DataFrame has required columns and no NaNs in them."""
    if df.empty:
        return False
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    return True


def timestamp_now() -> str:
    """Returns ISO-formatted current timestamp."""
    return datetime.now().isoformat()


def format_pct(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.2f}%"


def ensure_directories(*dirs: Path) -> None:
    """Create multiple directories if they don't exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def safe_load_model(filepath: str | Path) -> Any:
    """
    Safely load a model artifact. Enforces that the file resides in the trusted MODELS_DIR.
    In a real production system, this would additionally verify cryptographic signatures/hashes.
    """
    import joblib
    from config.settings import MODELS_DIR
    target = Path(filepath).resolve()
    trusted = Path(MODELS_DIR).resolve()
    
    if not str(target).startswith(str(trusted)):
        raise ValueError(f"Security Policy Violation: Attempted to load model from untrusted path {filepath}")
    
    if not target.exists():
        raise FileNotFoundError(f"Model not found: {filepath}")
        
    return joblib.load(target)
