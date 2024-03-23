import numpy as np
import torch
import cv2


def dilate_mask(mask, kernel=15):
    """
    :param masks: masks to dilate
    :return: dilated masks
    """
    device = mask.device
    kernel = np.ones((kernel, kernel), np.uint8)

    dilated_mask = torch.zeros_like(mask)
    # print("size of dilated mask before numpying", dilated_mask.shape)
    mask = mask[0].detach().cpu().numpy()
    if np.sum(mask) < 1:
        dilated_mask = mask
        return dilated_mask

    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    dilated_mask = torch.from_numpy(dilated_mask).to(device).unsqueeze(0)

    return dilated_mask


def binarize_mask(mask, threshold=0.5):
    mask = torch.where(mask > threshold, torch.ones_like(mask), torch.zeros_like(mask))
    return mask
