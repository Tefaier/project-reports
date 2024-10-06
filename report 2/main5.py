import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import cv2
from torchvision import transforms, utils

# pseudo dataset
class NumbersDataset(Dataset):
    def __init__(self, low, high):
        self.samples = list(range(low, high))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

dataset = NumbersDataset(0, 1000)
dataLoader = DataLoader(dataset, batch_size=40, shuffle=True)
# can be enumerated to go through all batches created based on dataset

# from torch documentation but cv2 put into image reading
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# list of transformation function to apply on image
# torchvision has many transforms including random versions
dataTransform = [
    transforms.ToPILImage(),
    transforms.ToTensor()
]
# combines them into one function
transform = transforms.Compose(dataTransform)

dataset = CustomImageDataset("", "", transform)

# split on data zones
train, val, test = torch.utils.data.random_split(dataset, [80, 10, 10])