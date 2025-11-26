"""Loading config from yaml, and file creation"""
from pathlib import Path
import yaml
from utils.logger import get_logger

logger = get_logger(__name__)


def load_config():
    "Loading paths, and cofiguration from yaml"
    path = "/opt/airflow/config/config.yml"
    logger.info(f"Loading configuration from {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = Path(config["data"]["BASE_DIR"])
    raw_dir = Path(config["data"]["RAW_DIR"])
    mask_dir = Path(config["data"]["MASK_DIR"])
    meta_dir = Path(config["data"]["META_DIR"])
    index_csv = Path(config["data"]["INDEX_FILE"])
    logger.debug(f"Parsed directories: {raw_dir}, {mask_dir}, {meta_dir}")

    for d in [raw_dir, mask_dir, meta_dir]:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {d}")

    logger.info("Configuration loaded successfully ✅")
    return {
        "base_dir": str(base_dir),
        "raw_dir": str(raw_dir),
        "mask_dir": str(mask_dir),
        "meta_dir": str(meta_dir),
        "index_csv": str(index_csv)
    }
