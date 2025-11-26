"""Generation of the masks"""
import pandas as pd
from pathlib import Path
import os
import numpy as np
import imageio.v2 as imageio
from tasks.segmentation.segmentations import segment_cytoplasm_with_nuclei, segment_nucleus, segment_protein


# Input csv_path
def generate_masks(**kwargs):
    """Generation of the masks"""
    ti = kwargs["ti"]
    cfg = ti.xcom_pull(task_ids="load_config")
    index_csv = Path(cfg["index_csv"])
    df = pd.read_csv(index_csv)

    for i, row in df.iterrows():
        print(f"Segmenting {row['ens_id']} - {row['sample_id']}")

        nuc_img = imageio.imread(row["nucleus_path"])
        cyt_img = imageio.imread(row["cytoplasm_path"])
        prot_img = imageio.imread(row["protein_path"])

        nuc_mask = segment_nucleus(nuc_img)
        cyt_mask = segment_cytoplasm_with_nuclei(cyt_img, nuc_mask)
        prot_mask = segment_protein(prot_img)

        # Binarize cytoplasm (all >0 → 1)
        cyt_mask_bin = cyt_mask > 0

        combined_mask = np.zeros_like(nuc_mask, dtype=np.uint8)

        # Assign in order of priority
        combined_mask[cyt_mask_bin] = 2   # cytoplasm
        combined_mask[nuc_mask > 0] = 1   # nucleus
        combined_mask[prot_mask > 0] = 3  # protein

        os.makedirs(os.path.dirname(row["mask_path"]), exist_ok=True)
        imageio.imwrite(row["mask_path"], combined_mask.astype(np.uint8))
