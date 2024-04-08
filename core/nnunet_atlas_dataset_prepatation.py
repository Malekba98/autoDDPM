import torch

image_dir = ["./data/ATLAS/splits/atlas_train_png.csv"]
label_dir = ["./data/ATLAS/splits/atlas_train_mask_png.csv"]

mask_dir = ["./data/ATLAS/splits/atlas_train_brain_mask_png.csv"]

from torch.utils.data import DataLoader

from data.loaders.ixi_loader import mask_preprocessing_loader
import csv
import os
import shutil

from PIL import Image 
import numpy as np

PREPROCESS_TRAIN = False


device = "cuda" if torch.cuda.is_available() else "cpu"

if PREPROCESS_TRAIN:
    dataset = mask_preprocessing_loader(
    image_dir,
    label_dir=label_dir,
    mask_dir=mask_dir,
    target_size=(128, 128),
    test=False,
    )

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True
    )

    for data in loader:
        image = data[0].to(device)
        mask = data[1].to(device)
        idx = data[2]
        filename = data[3]
        mask_filename = data[4]
        # copy file with filename to new directory "./nnunet_data/nnunet_raw/Dataset500_ATLAS/imagesTr"
        # copy file with mask_filename to new directory "./nnunet_data/nnunet_raw/Dataset500_ATLAS/labelsTr"
        # ATLAS_001_0000.png
        image_dest = f"./nnunet_data/nnunet_raw/Dataset500_ATLAS/imagesTr/ATLAS_{int(idx):03d}_0000.png"
        mask_dest = f"./nnunet_data/nnunet_raw/Dataset500_ATLAS/labelsTr/ATLAS_{int(idx):03d}.png"
        shutil.copy(filename[0], image_dest)
        #shutil.copy(mask_filename[0], mask_dest)

        mask_array = np.array(Image.open(mask_filename[0]))

        # Convert 255 to 1
        mask_array[mask_array == 255] = 1

        # Save the converted mask back to a file
        Image.fromarray(mask_array).save(mask_dest)
else:
    image_dir = ["./data/ATLAS/splits/atlas_test_png.csv"]
    label_dir = ["./data/ATLAS/splits/atlas_test_png.csv"]

    mask_dir = ["./data/ATLAS/splits/atlas_test_png.csv"]

    dataset = mask_preprocessing_loader(
    image_dir,
    label_dir=label_dir,
    mask_dir=mask_dir,
    target_size=(128, 128),
    test=False,
    )

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True
    )

    for data in loader:
        filename = data[3]
        idx = data[2]

        image_dest = f"./nnunet_data/nnunet_raw/Dataset500_ATLAS/imagesTs/ATLAS_{int(idx):03d}_0000.png"
        shutil.copy(filename[0], image_dest)