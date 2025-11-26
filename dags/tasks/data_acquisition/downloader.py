"""Downloading all the images depending on the ens id"""
import requests
from utils.logger import get_logger

logger = get_logger(__name__)


def download_file(url, save_path):
    """Download image if not exists"""
    logger.info("Starting to download files")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not save_path.exists():
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(r.content)
        else:
            print(f"[!] Failed download {url}")
    logger.info("Finish downloading all the files")
