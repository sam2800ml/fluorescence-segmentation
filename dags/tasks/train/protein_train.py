

import torch
from tasks.models.protein_unet_model import NestedUNet
from tasks.train.train_model import train_unet_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


protein_model = train_unet_model(
    device=device,
    model=NestedUNet,
    in_channels=3,
    out_channels=4,
    learning_rate=1e-3,
    num_epochs=1000,
    patience=20,
    train_loader=train_loader_protein,
    val_loader=val_loader_protein,
    experiment_name="ProteinSegmentation",
    run_name="nestedunet_protein",
    protein=True,
    model_name="nested_unet_protein_model_.pth"
)