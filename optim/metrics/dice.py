import numpy as np
import torch


def compute_dice(predictions, gt, th):
    predictions[predictions > th] = 1
    predictions[predictions < 1] = 0
    eps = 1e-6
    # flatten label and prediction tensors
    inputs = predictions.flatten()
    targets = gt.flatten()

    intersection = (inputs * targets).sum()
    dice = (2.0 * intersection) / (inputs.sum() + targets.sum() + eps)
    return dice


def dice_coefficient_batch(a, b):
    intersection = np.sum(a * b, axis=(1, 2, 3))
    union = np.sum(a, axis=(1, 2, 3)) + np.sum(b, axis=(1, 2, 3))
    dice = (2.0 * intersection) / union
    mean_dice = np.mean(dice)
    return mean_dice


def dice_coefficient_batch_(a, b):
    intersection = torch.sum(a * b, dim=(1, 2, 3))
    union = torch.sum(a, dim=(1, 2, 3)) + torch.sum(b, dim=(1, 2, 3))
    dice = (2.0 * intersection) / union
    return dice
