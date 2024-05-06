import torch

image_dir = ["./data/ATLAS/splits/atlas_train_png.csv"]
label_dir = ["./data/ATLAS/splits/atlas_train_mask_png.csv"]

mask_dir = ["./data/ATLAS/splits/atlas_train_brain_mask_png.csv"]

from torch.utils.data import DataLoader

from data.loaders.ixi_loader import mask_preprocessing_loader
import csv
import os
import numpy as np

types_of_pathos = {"low": 70,"medium": 100,"high": 150}

PIXELS_THRESHOLD = 1
dataset = mask_preprocessing_loader(
    image_dir,
    label_dir=label_dir,
    mask_dir=mask_dir,
    target_size=(128, 128),
    test=False,
)
loader = DataLoader(
    dataset, batch_size=1, shuffle=True, drop_last=False, pin_memory=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"

all_pixel_numbers = []
for data in loader:
    image = data[0].to(device)
    mask = data[1].to(device)
    filename = data[3]
    mask_filename = data[4]
    brain_mask_filename = data[5]
    non_zero_pixels = mask.sum()
    if non_zero_pixels.item() >= PIXELS_THRESHOLD:
        all_pixel_numbers.append(int(non_zero_pixels.item()))
    #if non_zero_pixels.item() >= PIXELS_THRESHOLD:
    #    print("filename", filename[0])
    #    print("mask_filename", mask_filename[0])
    #    print("brain_mask_filename", brain_mask_filename[0])

print("all_pixel_numbers", all_pixel_numbers)

all_pixel_numbers = np.array(all_pixel_numbers)
higher_25_percentile = np.percentile(all_pixel_numbers, 75)
lower_25_percentile = np.percentile(all_pixel_numbers, 25)
between_25_and_75 = np.percentile(all_pixel_numbers, [25, 75])


types_of_pathos["low"] = lower_25_percentile
types_of_pathos["high"] = higher_25_percentile



print("Higher 25 percentile:", higher_25_percentile)
print("Lower 25 percentile:", lower_25_percentile)
print("Between 25 and 75 percentile:", between_25_and_75)


labels = []
for data in loader:
    image = data[0].to(device)
    mask = data[1].to(device)
    filename = data[3]
    mask_filename = data[4]
    brain_mask_filename = data[5]
    non_zero_pixels = mask.sum()
    if non_zero_pixels.item() <= 0:
        continue
    if non_zero_pixels.item() >= types_of_pathos["high"]:
        labels.append("high")
    elif non_zero_pixels.item() <= types_of_pathos["low"]:
        labels.append("low")
    else:
        labels.append("medium")


high_count = labels.count("high")
low_count = labels.count("low")
medium_count = labels.count("medium")

print("Number of high:", high_count)
print("Number of low:", low_count)
print("Number of medium:", medium_count)


print("number of cases", len(labels))