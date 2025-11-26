""" Logger """
import logging
from pathlib import Path
import os


def get_logger(name: str = "cell_segmentation", log_file: str = "logs/project.log"):
    """Create and configure a reusable logger."""
    # Use AIRFLOW_HOME environment variable or fallback to /opt/airflow
    base_log_dir = Path(os.getenv("AIRFLOW_HOME", "/opt/airflow")) / "logs"
    base_log_dir.mkdir(parents=True, exist_ok=True)
    # Construct the full log file path
    full_log_file = base_log_dir / Path(log_file).name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler(full_log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger
