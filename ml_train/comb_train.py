
from train_model import train_unet_model
from protein_unet_model import AttentionUNet
from dataset_loader import CellSegmentationDataset
from torch.utils.data import DataLoader, random_split
from transformations import build_transforms
import argparse
import torch
import os
import psutil


def train_model_t(csv_path):

    print(f"Starting training with CSV {csv_path}")

    transform = build_transforms()

    complete_dataset = CellSegmentationDataset(csv_path, transform=transform, use_protein=False)

    train_size = int(0.7 * len(complete_dataset))
    val_size = int(0.2 * len(complete_dataset))
    test_size = len(complete_dataset) - train_size - val_size

    train_set, val_set, _ = random_split(
        complete_dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    best_model = train_unet_model(
        device=device,
        model=AttentionUNet,
        in_channels=3,
        out_channels=4,
        learning_rate=1e-3,
        num_epochs=100,
        patience=10,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name="ProteinSegmentation",
        run_name="unet_comb_model",
        model_name="unet_comb_model.pth"
    ) 

    return "training_completed"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the dataset CSV")
    args = parser.parse_args()
    
    train_model_t(args.csv_path)