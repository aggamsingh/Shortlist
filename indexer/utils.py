import os
import hashlib
import json
import logging

def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with a standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = get_logger("indexer.utils")

def calculate_file_hash(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        raise

def load_index_state(state_path: str) -> dict:
    """Load indexing state from a JSON file."""
    if not os.path.exists(state_path):
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load index state from {state_path}, returning empty state: {e}")
        return {}

def save_index_state(state_path: str, state: dict) -> None:
    """Save indexing state to a JSON file."""
    try:
        dir_name = os.path.dirname(state_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save index state to {state_path}: {e}")
