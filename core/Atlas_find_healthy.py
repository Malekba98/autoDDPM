import torch

image_dir = ["./data/ATLAS/splits/atlas_train_png.csv"]
label_dir = ["./data/ATLAS/splits/atlas_train_mask_png.csv"]

mask_dir = ["./data/ATLAS/splits/atlas_train_brain_mask_png.csv"]

from torch.utils.data import DataLoader

from data.loaders.ixi_loader import mask_preprocessing_loader
import csv
import os

PIXELS_THRESHOLD = 0
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
sum = 0
if os.path.exists(f"Atlas_healthy.csv"):
    os.remove(f"Atlas_healthy.csv")

if os.path.exists(f"Atlas_mask_healthy.csv"):
    os.remove(f"Atlas_mask_healthy.csv")

if os.path.exists(f"Atlas_brain_mask_healthy.csv"):
    os.remove(f"Atlas_brain_mask_healthy.csv")

for data in loader:
    image = data[0].to(device)
    mask = data[1].to(device)
    filename = data[3]
    mask_filename = data[4]
    brain_mask_filename = data[5]
    non_zero_pixels = mask.sum()
    if non_zero_pixels.item() <= PIXELS_THRESHOLD:
        print("filename", filename[0])
        print("mask_filename", mask_filename[0])
        print("brain_mask_filename", brain_mask_filename[0])

        with open("Atlas_brain_mask_healthy.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([brain_mask_filename[0]])

        with open("Atlas_mask_healthy.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([mask_filename[0]])

        with open("Atlas_healthy.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([filename[0]])
        sum += 1


print("number of healthy samples", sum)
