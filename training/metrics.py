# training/metrics.py

import torch


def dice_score(preds, targets, threshold=0.5):

    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()

    dice = (2. * intersection) / (
        preds.sum() + targets.sum() + 1e-8
    )

    return dice.item()
