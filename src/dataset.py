import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class LiverTumorDataset(Dataset):
    def __init__(self, txt_file, image_size=(256, 256)):
        self.pairs = []
        self.image_size = image_size

        with open(txt_file, 'r') as f:
            for line in f:
                image_path, mask_path = line.strip().split(',')
                self.pairs.append((image_path, mask_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise FileNotFoundError(f"Missing file: {img_path} or {mask_path}")

        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image), torch.tensor(mask)
