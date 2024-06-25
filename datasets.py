import os
from typing import List, Dict, Tuple

from PIL import Image

import torch
import torch.utils
import torchvision
from torch.utils.data import Dataset, DataLoader


class DogHeartLabeledDataset(Dataset):

    def __init__(self, data_root: str) -> None:
        self.data_root: str = data_root
        self.classes: List[str] = os.listdir(data_root)
        self.class_to_idx: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.transformation = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])

        self.filenames: List[str] = []
        self.filepaths: List[str] = []
        self.labels: List[int] = []

        for class_name in self.classes:
            path: str = os.path.join(data_root, class_name)
            for filename in os.listdir(path):
                self.filenames.append(filename)
                self.filepaths.append(os.path.join(path, filename))
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        filename: str = self.filenames[idx]
        filepath: str = self.filepaths[idx]
        image: Image = Image.open(filepath)
        label: torch.Tensor = torch.tensor(self.labels[idx])
        tensor: torch.Tensor = self.transformation(image)
        tensor = tensor.squeeze(0)
        return tensor, label, filename
    

class DogHearUnlabeledDataset(Dataset):

    def __init__(self, data_root: str) -> None:
        self.data_root: str = data_root
        self.transformation = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
        ])
        self.filenames: List[str] = os.listdir(self.data_root)
    
    def __len__(self) -> int:
        return len(self.filenames)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        filename: str = self.filenames[idx]
        image: Image = Image.open(os.path.join(self.data_root, filename))
        tensor: torch.Tensor = self.transformation(image)
        tensor = tensor.squeeze(0)
        return tensor, filename


if __name__ == '__main__':

    # P1.Q1

    train_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Train')
    valid_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Valid')
    test_dataset = DogHearUnlabeledDataset(data_root='Test')

    train_dataloader: DataLoader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    valid_dataloader: DataLoader = DataLoader(dataset=valid_dataset, batch_size=32, shuffle=False)
    test_dataloader: DataLoader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)









