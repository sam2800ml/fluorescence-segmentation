""" Dataset builder"""
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from pathlib import Path
import yaml
import pandas as pd
import csv
import os
from tasks.data_acquisition.hpa_parser import get_image_urls
from tasks.data_acquisition.downloader import download_file
from utils.logger import get_logger


logger = get_logger(__name__)


def build_dataset(**kwargs):
    """Building the dataset"""
    ti = kwargs["ti"]
    cfg = ti.xcom_pull(task_ids="load_config")  # ✅ correct key

    raw_dir = Path(cfg["raw_dir"])
    mask_dir = Path(cfg["mask_dir"])
    index_csv = Path(cfg["index_csv"])

    txt_file = Path("/opt/airflow/ends_id.txt")
    ens_ids = [line.strip() for line in open(txt_file)]

    logger.info("Starting building dataset")

    if index_csv.exists():
        existing = pd.read_csv(index_csv)
        downloaded = set(existing["ens_id"].unique())
    else:
        downloaded = set()

    with open(index_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not index_csv.exists() or index_csv.stat().st_size == 0:
            writer.writerow(["ens_id", "sample_id", "nucleus_path", "cytoplasm_path",
                             "protein_path", "comb_path", "mask_path"])

        for ens_id in ens_ids:
            if ens_id in downloaded:
                logger.info(f"✅ {ens_id} already processed — skipping.")
                continue

            urls = get_image_urls(ens_id)
            for i, (nuc, cyt, prot, comb) in enumerate(urls, start=1):
                sample_id = f"sample_{i:03d}"
                sample_raw = raw_dir / ens_id / sample_id
                sample_mask = mask_dir / ens_id / sample_id

                nuc_path = sample_raw / "nucleus.jpg"
                cyt_path = sample_raw / "cytoplasm.jpg"
                prot_path = sample_raw / "protein.jpg"
                comb_path = sample_raw / "combined.jpg"
                mask_path = sample_mask / "mask.png"

                download_file(nuc, nuc_path)
                download_file(cyt, cyt_path)
                download_file(prot, prot_path)
                download_file(comb, comb_path)

                writer.writerow([
                    ens_id, sample_id,
                    str(nuc_path), str(cyt_path), str(prot_path), str(comb_path), str(mask_path)
                ])
    logger.info("Finish building the dataset")
