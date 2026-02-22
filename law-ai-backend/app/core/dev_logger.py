"""
Specialized logger for development data tracing.
"""
import logging
from logging.handlers import RotatingFileHandler
import json
from pathlib import Path
from typing import Any

# Create a logger
dev_data_logger = logging.getLogger("DevDataTrace")
dev_data_logger.setLevel(logging.INFO)
dev_data_logger.propagate = False  # Prevent passing logs to the root logger

# Create a handler
# Use RotatingFileHandler to prevent log files from growing indefinitely
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_LOG_PATH = _BACKEND_ROOT / "dev_data_trace.log"

handler = RotatingFileHandler(
    str(_LOG_PATH),
    maxBytes=10 * 1024 * 1024,
    backupCount=5,  # 10MB per file, 5 backups
    encoding="utf-8",
)
handler.setLevel(logging.INFO)

# Create a formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
if not dev_data_logger.handlers:
    dev_data_logger.addHandler(handler)

def _dev_trace_enabled() -> bool:
    """Central switch so callers don't need to duplicate env checks."""
    try:
        # Local import to avoid accidental import cycles.
        from app.core.config import Config

        return str(getattr(Config, "ENV", "")).lower() in {"dev", "development"}
    except Exception:
        return False


def format_and_log(endpoint: str, action: str, data_name: str, data_content: Any) -> None:
    """
    Formats the data as a pretty-printed JSON string and logs it.
    """
    if not _dev_trace_enabled():
        return

    try:
        # Use a default function to handle non-serializable objects if necessary
        data_str = json.dumps(data_content, indent=2, default=str)
    except TypeError:
        data_str = str(data_content) # Fallback to simple string representation

    message = (
        f"Endpoint: {endpoint}\n"
        f"Action: {action}\n"
        f"Data Name: {data_name}\n"
        f"Data Content:\n{data_str}\n"
        f"--------------------------------------------------"
    )
    dev_data_logger.info(message)
