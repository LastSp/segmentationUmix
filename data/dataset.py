import cv2

import numpy as np
import torch
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, is_val):

        self.images_path = images_path
        self.masks_path = masks_path
        self._is_val = is_val
        self.n_samples = len(images_path)

    def _random_flip(self, image, mask):
        random_flip = np.random.randint(-1, 2)

        image = cv2.flip(image, random_flip)
        mask = cv2.flip(mask, random_flip)

        return image, mask

    def __getitem__(self, index):
        is_augment = np.random.random() if not self._is_val else 0

        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / 255.0

        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0

        if is_augment > 0.5:
            image, mask = self._random_flip(image, mask)

        if len(mask.shape) != 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)

        return image, mask

    def __len__(self):
        return self.n_samples