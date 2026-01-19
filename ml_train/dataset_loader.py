""" Cell segmentation"""
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class CellSegmentationDataset(Dataset):
    def __init__(self, csv_path, transform=None, use_protein=False):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.use_protein = use_protein

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        img_path = row["protein_path"] if self.use_protein else row["comb_path"]

        # Load combined RGB image
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256))

        # Load combined mask (with class labels 0..3)
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_UNCHANGED)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.int64)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()  # [H,W,C] to [C,H,W]
        mask = torch.from_numpy(mask).long()
        return image, mask
