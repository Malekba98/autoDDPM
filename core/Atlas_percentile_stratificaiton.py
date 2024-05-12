import torch
from sklearn.model_selection import train_test_split

image_dir = ["./data/ATLAS/splits/atlas_train_png.csv"]
label_dir = ["./data/ATLAS/splits/atlas_train_mask_png.csv"]

mask_dir = ["./data/ATLAS/splits/atlas_train_brain_mask_png.csv"]

from torch.utils.data import DataLoader

from data.loaders.ixi_loader import mask_preprocessing_loader
import csv
import os
import numpy as np

types_of_pathos = {"low": 70, "medium": 100, "high": 150}

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
    # if non_zero_pixels.item() >= PIXELS_THRESHOLD:
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
file_paths = []
mask_file_paths = []
brain_mask_file_paths = []


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

    file_paths.append([filename[0]])
    mask_file_paths.append([mask_filename[0]])
    brain_mask_file_paths.append([brain_mask_filename[0]])


high_count = labels.count("high")
low_count = labels.count("low")
medium_count = labels.count("medium")

print("Number of high:", high_count)
print("Number of low:", low_count)
print("Number of medium:", medium_count)


print("number of cases", len(labels))

# Split data into train, val, and test sets
train_ratio = 0.78
val_ratio = 0.1
test_ratio = 0.12

train_files, temp_files, train_labels, temp_labels, train_mask_files, temp_mask_files, train_brain_mask_files, temp_brain_mask_files = train_test_split(
    file_paths,
    labels,
    mask_file_paths,
    brain_mask_file_paths,
    random_state=42,
    test_size=(1 - train_ratio),
    stratify=labels,
)

val_files, test_files, val_labels, test_labels, val_mask_files, test_mask_files, val_brain_mask_files, test_brain_mask_files = train_test_split(
    temp_files,
    temp_labels,
    temp_mask_files,
    temp_brain_mask_files,
    random_state=42,
    test_size=test_ratio / (val_ratio + test_ratio),
    stratify=temp_labels,
)

#print("train files", train_files)

#print("val files", val_files)
#print("test files", test_files)


train_file = "atlas_train_png.csv"
val_file = "atlas_val_png.csv"

test_file = "atlas_test_png.csv"

output_folder = f"data/ATLAS/splits_over_{PIXELS_THRESHOLD}_stratified_all_splits"


os.makedirs(output_folder, exist_ok=True)

train_file_path = os.path.join(output_folder, train_file)
val_file_path = os.path.join(output_folder, val_file)

test_file_path = os.path.join(output_folder, test_file)

with open(train_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(train_files)


with open(val_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(val_files)


with open(test_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(test_files)

print("Train file saved at:", train_file_path)

print("Validation file saved at:", val_file_path)

print("Test file saved at:", test_file_path)

# DO SAME FOR MASKS


train_mask_file = "atlas_train_mask_png.csv"
val_mask_file = "atlas_val_mask_png.csv"

test_mask_file = "atlas_test_mask_png.csv"

train_file_path = os.path.join(output_folder, train_mask_file)
val_file_path = os.path.join(output_folder, val_mask_file)

test_file_path = os.path.join(output_folder, test_mask_file)

with open(train_file_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerows(train_mask_files)

with open(val_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(val_mask_files)

with open(test_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(test_mask_files)

print("Train mask file saved at:", train_file_path)
print("Validation mask file saved at:", val_file_path)
print("Test mask file saved at:", test_file_path)

# DO SAME FOR BRAIN MASKS

train_brain_mask_file = "atlas_train_brain_mask_png.csv"
val_brain_mask_file = "atlas_val_brain_mask_png.csv"
test_brain_mask_file = "atlas_test_brain_mask_png.csv"

train_file_path = os.path.join(output_folder, train_brain_mask_file)
val_file_path = os.path.join(output_folder, val_brain_mask_file)
test_file_path = os.path.join(output_folder, test_brain_mask_file)

with open(train_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(train_brain_mask_files)

with open(val_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(val_brain_mask_files)

with open(test_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(test_brain_mask_files)

print("Train brain mask file saved at:", train_file_path)
print("Validation brain mask file saved at:", val_file_path)
print("Test brain mask file saved at:", test_file_path)