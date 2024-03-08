import torch
image_dir = ['./data/ATLAS/splits/atlas_train_png.csv']
mask_dir = ['./data/ATLAS/splits/atlas_train_mask_png.csv']

from torch.utils.data import DataLoader

from data.loaders.ixi_loader import mask_preprocessing_loader
import csv
import os
PIXELS_THRESHOLD = 100
dataset = mask_preprocessing_loader(image_dir, label_dir=mask_dir, target_size=(128, 128), test=False)
loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sum = 0
if os.path.exists('Atlas_over_100.csv'):
    os.remove('Atlas_over_100.csv')

if os.path.exists('Atlas_mask_over_100.csv'):
    os.remove('Atlas_mask_over_100.csv')

for data in loader:
    image = data[0].to(device)
    mask = data[1].to(device)
    filename = data[3]
    mask_filename = data[4]
    non_zero_pixels = mask.sum()
    if non_zero_pixels.item() >= PIXELS_THRESHOLD:
        print("filename", filename[0])
        print("mask_filename", mask_filename[0])
        with open('Atlas_mask_over_100.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([mask_filename[0]])

        with open('Atlas_over_100.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([filename[0]])
        sum += 1

train_ratio = 0.8
train_size = int(train_ratio * sum)
train_file = 'atlas_train_png.csv'
val_file = 'atlas_val_png.csv'

with open('Atlas_over_100.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

train_data = data[:train_size]
val_data = data[train_size:]

output_folder = 'data/ATLAS/splits_over_100'
os.makedirs(output_folder, exist_ok=True)

train_file_path = os.path.join(output_folder, train_file)
val_file_path = os.path.join(output_folder, val_file)

with open(train_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(train_data)

with open(val_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(val_data)


print("Train file saved at:", train_file_path)
print("Validation file saved at:", val_file_path)

# do same for masks 

with open('Atlas_mask_over_100.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)

train_data = data[:train_size]
val_data = data[train_size:]

mask_train_file = 'atlas_train_mask_png.csv'
mask_val_file = 'atlas_val_mask_png.csv'
train_file_path = os.path.join(output_folder, mask_train_file)
val_file_path = os.path.join(output_folder, mask_val_file)

with open(train_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(train_data)

with open(val_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(val_data)

print("mask Train file saved at:", train_file_path)
print("mask Validation file saved at:", val_file_path)

print("number of over 100 pixels", sum)
