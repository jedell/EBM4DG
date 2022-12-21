# create 5 files in each of the lowest level directories (pa, lateral)

import os

import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
import random
import numpy as np
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ChestXRayDataset(Dataset):
    def __init__(self, root_dir_1, root_dir_2, category, projection, transform=None, device='cpu'):
        self.root_dir_1 = root_dir_1
        self.root_dir_2 = root_dir_2
        self.category = category
        self.projection = projection
        self.transform = transform
        self.device = device

        self.samples = []
        category_dir = os.path.join(self.root_dir_1, self.category)
        projection_dir = os.path.join(category_dir, self.projection)
        for file in os.listdir(projection_dir):
            self.samples.append((file, 1, 0))

        if root_dir_2:
            category_dir = os.path.join(self.root_dir_2, self.category)
            projection_dir = os.path.join(category_dir, self.projection)
            for file in os.listdir(projection_dir):
                self.samples.append((file, 1, 1))

        negative_dir = os.path.join(self.root_dir_1, "N", self.projection)
        for file in os.listdir(negative_dir):
            self.samples.append((file, 0, 0))

        if root_dir_2:
            negative_dir = os.path.join(self.root_dir_2, "N", self.projection)
            for file in os.listdir(negative_dir):
                self.samples.append((file, 0, 1))

        # random.shuffle(self.samples)
        print(projection_dir)
        print(negative_dir)
        # print(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # r = random.random()
        # if r >= 0.1:
        #     idx += 1
        file, label, dir = self.samples[idx]
        if label == 1:
            file_path = os.path.join(
                self.root_dir_1 if dir == 0 else self.root_dir_2, self.category, self.projection, file
            )
        else:
            file_path = os.path.join(self.root_dir_1 if dir == 0 else self.root_dir_2, "N", self.projection, file)

        # return file_path, label, dir

        # image = torchvision.io.read_image(file_path) # read image as torch tensor
        # read image as PIL image
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        print(image.shape)
        image = np.transpose(image, (2, 0, 1)).astype("float64")
        image *= (1/image.max())
        image = torch.tensor(image, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)

        return image, label, dir


def get_data_loader(root_dir, category, projection, batch_size, transform=None):
    dataset = ChestXRayDataset(root_dir, category, projection, transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return data_loader


if __name__ == "__main__":
    print("Testing dataloader.py:\nroot_dir = ./padchest\ncategory = A\nprojection = pa\nbatch_size = 4\ntransform = None\n")
    root_dir = "./padchest"
    category = "A"
    projection = "pa"
    batch_size = 4
    # transform all images to 224x224
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ]
    )

    # remove .DS_Store files from root_dir/category/projection and root_dir/N/projection
    category_dir = os.path.join(root_dir, category)
    projection_dir = os.path.join(category_dir, projection)
    for file in os.listdir(projection_dir):
        if file == ".DS_Store":
            os.remove(os.path.join(projection_dir, file))
    negative_dir = os.path.join(root_dir, "N", projection)
    for file in os.listdir(negative_dir):
        if file == ".DS_Store":
            os.remove(os.path.join(negative_dir, file))

    data_loader = get_data_loader(
        root_dir, category, projection, batch_size, transform
    )
    for batch in data_loader:
        images, labels = batch
        print(images.shape, labels.shape)
        break
