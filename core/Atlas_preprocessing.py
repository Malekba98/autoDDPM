import torch
image_dir = ['./data/ATLAS/splits/atlas_train_png.csv']
mask_dir = ['./data/ATLAS/splits/atlas_train_mask_png.csv']

from torch.utils.data import DataLoader

from data.loaders.ixi_loader import mask_preprocessing_loader
import csv
PIXELS_THRESHOLD = 100
dataset = mask_preprocessing_loader(image_dir, label_dir=mask_dir, target_size=(128, 128), test=False)
loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            pin_memory=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sum = 0
for data in loader:
    image = data[0].to(device)
    mask = data[1].to(device)
    index = data[2].to(device)
    filename = data[3]
    non_zero_pixels = mask.sum()
    if non_zero_pixels.item() >= PIXELS_THRESHOLD:
        print("filename", filename[0])
        with open('Atlas_over_100.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([filename[0]])
        sum += 1
    #print(image.shape, mask.shape)
    
    #print(index)

print("number of over 100 pixels", sum)