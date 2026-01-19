
import torch
import torch.optim as optim
import numpy as np
import random
import copy
import mlflow
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, jaccard_score


def compute_metrics(y_pred, y_true):
    # y_pred shape: (N, C, H, W); y_true shape: (N, 1, H, W)
    y_pred_labels = torch.argmax(y_pred, dim=1).cpu().numpy().astype(np.uint8)  # (N, H, W)
    y_true_labels = y_true.squeeze(1).cpu().numpy().astype(np.uint8)  # (N, H, W)

    # flatten
    preds_flat = y_pred_labels.flatten()
    truths_flat = y_true_labels.flatten()

    # Dice coefficient per class approximation (macro average)
    classes = np.unique(truths_flat)
    dice_scores = []
    for cls in classes:
        pred_cls = (preds_flat == cls).astype(int)
        true_cls = (truths_flat == cls).astype(int)
        intersection = np.sum(pred_cls * true_cls)
        dice = (2 * intersection) / (np.sum(pred_cls) + np.sum(true_cls) + 1e-6)
        dice_scores.append(dice)
    dice = np.mean(dice_scores)

    jaccard = jaccard_score(truths_flat, preds_flat, average='macro')
    prec = precision_score(truths_flat, preds_flat, average='macro', zero_division=0)
    rec = recall_score(truths_flat, preds_flat, average='macro', zero_division=0)

    return dice, jaccard, prec, rec
