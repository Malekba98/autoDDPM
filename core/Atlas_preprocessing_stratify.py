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
if os.path.exists(f"Atlas_over_{PIXELS_THRESHOLD}.csv"):
    os.remove(f"Atlas_over_{PIXELS_THRESHOLD}.csv")

if os.path.exists(f"Atlas_mask_over_{PIXELS_THRESHOLD}.csv"):
    os.remove(f"Atlas_mask_over_{PIXELS_THRESHOLD}.csv")

if os.path.exists(f"Atlas_brain_mask_over_{PIXELS_THRESHOLD}.csv"):
    os.remove(f"Atlas_brain_mask_over_{PIXELS_THRESHOLD}.csv")


from sklearn.model_selection import train_test_split

# Create a list to store the labels
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
    if non_zero_pixels.item() >= PIXELS_THRESHOLD:
        # Add 'unhealthy' to labels if mask size >= 1
        labels.append("unhealthy")
    else:
        # Add 'healthy' to labels if mask size = 0
        labels.append("healthy")
    file_paths.append([filename[0]])
    mask_file_paths.append([mask_filename[0]])
    brain_mask_file_paths.append([brain_mask_filename[0]])

# Split data into train and temp sets
(
    train_files,
    temp_files,
    train_labels,
    temp_labels,
    train_mask_files,
    temp_mask_files,
    train_brain_mask_files,
    temp_brain_mask_files,
) = train_test_split(
    file_paths,
    labels,
    mask_file_paths,
    brain_mask_file_paths,
    test_size=0.3,
    stratify=labels,
)

# Split temp data into validation and test sets
(
    val_files,
    test_files,
    val_labels,
    test_labels,
    val_mask_files,
    test_mask_files,
    val_brain_mask_files,
    test_brain_mask_files,
) = train_test_split(
    temp_files,
    temp_labels,
    temp_mask_files,
    temp_brain_mask_files,
    test_size=0.5,
    stratify=temp_labels,
)

# Now you have stratified train, validation, and test sets

print("train files", train_files)

train_file = "atlas_train_png.csv"
val_file = "atlas_val_png.csv"
test_file = "atlas_test_png.csv"


output_folder = f"data/ATLAS/splits_over_{PIXELS_THRESHOLD}"
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

# do same for masks

train_file = "atlas_train_mask_png.csv"
val_file = "atlas_val_mask_png.csv"
test_file = "atlas_test_mask_png.csv"


os.makedirs(output_folder, exist_ok=True)

train_file_path = os.path.join(output_folder, train_file)
val_file_path = os.path.join(output_folder, val_file)
test_file_path = os.path.join(output_folder, test_file)

with open(train_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(train_mask_files)

with open(val_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(val_mask_files)

with open(test_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(test_mask_files)

print("mask Train file saved at:", train_file_path)
print("mask Validation file saved at:", val_file_path)

print("mask Test file saved at:", test_file_path)

# do same for brain mask

train_file = "atlas_train_brain_mask_png.csv"
val_file = "atlas_val_brain_mask_png.csv"
test_file = "atlas_test_brain_mask_png.csv"

os.makedirs(output_folder, exist_ok=True)

train_file_path = os.path.join(output_folder, train_file)
val_file_path = os.path.join(output_folder, val_file)
test_file_path = os.path.join(output_folder, test_file)

with open(train_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(train_brain_mask_files)

with open(val_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(val_brain_mask_files)

with open(test_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(test_brain_mask_files)

print("brain mask Train file saved at:", train_file_path)
print("brain mask Validation file saved at:", val_file_path)
print("brain mask Test file saved at:", test_file_path)




