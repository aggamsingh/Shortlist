import os
import time
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

def connect_qdrant(host: str, port: int, attempts: int = 10, delay: float = 2.0,
                   client_factory=None):
    """Connect to Qdrant, retrying while the server finishes starting.

    Under Docker Compose the database and the app start together, so the first
    connection usually races container startup. Retrying here is more reliable
    than gating on a compose healthcheck: it also covers a Qdrant restart while
    the API is already running, which a startup-only healthcheck never sees.
    """
    from qdrant_client import QdrantClient

    factory = client_factory or (lambda: QdrantClient(host=host, port=port))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            client = factory()
            client.get_collections()  # force a real round trip
            if attempt > 1:
                logger.info(f"Connected to Qdrant at {host}:{port} on attempt {attempt}.")
            return client
        except Exception as e:
            last_error = e
            if attempt < attempts:
                logger.warning(
                    f"Qdrant not reachable at {host}:{port} "
                    f"(attempt {attempt}/{attempts}): {e}. Retrying in {delay}s."
                )
                time.sleep(delay)
    raise ConnectionError(
        f"Could not reach Qdrant at {host}:{port} after {attempts} attempts: {last_error}"
    )
