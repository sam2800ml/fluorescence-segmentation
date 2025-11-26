def train_model_t(**kwargs):
    import torch, os, psutil, gc
    from tasks.train.train_model import train_unet_model
    from tasks.models.protein_unet_model import AttentionUNet
    from tasks.train.dataset_loader import CellSegmentationDataset
    from torch.utils.data import DataLoader, random_split
    from tasks.train.transformations import build_transforms

    # For debugging memory
    def log_mem(prefix=""):
        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / (1024**2)
        print(f"{prefix} — RAM usage: {rss_mb:.1f} MB")
        try:
            if torch.cuda.is_available():
                print("  GPU mem allocated (MB):",
                      torch.cuda.memory_allocated()/1024**2,
                      "| reserved:", torch.cuda.memory_reserved()/1024**2)
        except Exception:
            pass

    log_mem("Start task")

    ti = kwargs["ti"]
    meta = ti.xcom_pull(task_ids="prepare_datasets")
    csv_path = meta["csv_path"]

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

    log_mem("Before training")

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

    log_mem("After training")

    return "training_completed"
