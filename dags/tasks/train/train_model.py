import torch
import torch.optim as optim
from torch import amp
import numpy as np
import random
import copy
import mlflow
from tqdm import tqdm
from monai.losses import DiceCELoss
from tasks.train.metrics import compute_metrics
import os

os.environ["MLFLOW_TRACKING_URI"] = "file:/opt/airflow/mlruns"

def train_unet_model(
    device, model, in_channels, out_channels, learning_rate, num_epochs, patience,
    train_loader, val_loader, experiment_name, run_name, model_name, protein=False
):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    mlflow.set_tracking_uri("file:/opt/airflow/mlruns")

    use_amp = (device.type == "cuda")

    if use_amp:
        scaler = amp.GradScaler("cuda")
    else:
        scaler = None

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "in_channels": in_channels,
            "out_channels": out_channels,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "patience": patience,
            "protein": protein
        })

        model = model(in_channels=in_channels, num_classes=out_channels).to(device)
        criterion = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            reduction="mean",
        ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        best_val_loss = float("inf")
        wait = 0
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(num_epochs):
            model.train()
            train_running_loss = 0
            for idx, (img, mask) in enumerate(train_loader):
                img = img.float().to(device)
                mask = mask.long().unsqueeze(1).to(device)
                optimizer.zero_grad()
                # Use autocast if on GPU, else normal precision
                if use_amp:
                    with amp.autocast("cuda"):
                        y_pred = model(img)
                        loss = criterion(y_pred, mask)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    y_pred = model(img)
                    loss = criterion(y_pred, mask)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                train_running_loss += loss.item()
            train_loss = train_running_loss / (idx + 1)

            model.eval()
            val_running_loss = 0
            val_dice = 0
            val_jaccard = 0
            val_precision = 0
            val_recall = 0
            with torch.no_grad():
                for idx, (img, mask) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")):
                    img = img.float().to(device)
                    mask = mask.long().unsqueeze(1).to(device)
                    if use_amp:
                        with amp.autocast("cuda"):
                            y_pred = model(img)
                            loss = criterion(y_pred, mask)
                    else:
                        y_pred = model(img)
                        loss = criterion(y_pred, mask)
                    val_running_loss += loss.item()
                    dice, jaccard, prec, rec = compute_metrics(y_pred, mask)
                    val_dice += dice
                    val_jaccard += jaccard
                    val_precision += prec
                    val_recall += rec

            val_loss = val_running_loss / (idx + 1)
            val_dice /= (idx + 1)
            val_jaccard /= (idx + 1)
            val_precision /= (idx + 1)
            val_recall /= (idx + 1)

            scheduler.step(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_jaccard": val_jaccard,
                "val_precision": val_precision,
                "val_recall": val_recall,
            }, step=epoch)

            print("-" * 30)
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Dice: {val_dice:.4f} | Val Jaccard: {val_jaccard:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
            print("-" * 30)

            if val_loss < best_val_loss:
                print(f"✅ Validation loss improved from {best_val_loss:.4f} → {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1
                print(f"⚠️ No improvement for {wait} epochs.")
                if wait >= patience:
                    print("⏹️ Early stopping triggered.")
                    break

        model.load_state_dict(best_model_wts)

        mlflow.pytorch.log_model(model, artifact_path="model")
        print("Best model logged to MLflow.")

        torch.save(model.state_dict(), model_name)
        mlflow.log_artifact(model_name)

    return model
