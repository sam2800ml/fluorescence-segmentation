
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tasks.train.dataset_loader import CellSegmentationDataset
from tasks.train.transformations import build_transforms

def prepare_datasets(**kwargs):
    ti = kwargs["ti"]
    cfg = ti.xcom_pull(task_ids="load_config")
    csv_path = cfg["index_csv"]
    transform = build_transforms()

    protein_dataset = CellSegmentationDataset(csv_path, transform, use_protein=True)
    complete_dataset = CellSegmentationDataset(csv_path, transform, use_protein=False)

    return {
        "csv_path": csv_path,
        "protein_size": len(protein_dataset),
        "complete_size": len(complete_dataset)
    }
