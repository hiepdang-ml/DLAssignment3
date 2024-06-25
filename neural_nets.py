from typing import List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import DogHeartLabeledDataset, DogHearUnlabeledDataset
from feature_extractors import FeatureExtractor, HOG, SIFT


class NeuralNet(nn.Module):

    def __init__(self, n_hiddens: int, n_classes: int, feature_extractor: FeatureExtractor):
        super().__init__()
        self.n_hiddens: int = n_hiddens
        self.n_classes: int = n_classes
        self.feature_extractor: FeatureExtractor = feature_extractor
        self.fc1 = nn.LazyLinear(out_features=n_hiddens)
        self.fc2 = nn.LazyLinear(out_features=n_hiddens)
        self.fc3 = nn.LazyLinear(out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: np.ndarray = x.numpy()
        
        features: List[np.ndarray] = []
        for image in x:
            feature = self.feature_extractor(image_array=image)
            features.append(feature)
        
        features = torch.tensor(data=np.array(features), dtype=torch.float)
        y = torch.relu(self.fc1(input=features))
        y = torch.relu(self.fc2(input=y))
        y = torch.softmax(self.fc3(input=y), dim=1)
        return y

    def predict(self, test_dataloader: DataLoader) -> None:
        self.eval()

        filenames = []
        predictions = []
        with torch.no_grad():
            for images, fnames in test_dataloader:
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                filenames.extend(fnames)
                predictions.extend(predicted.numpy())

        prediction_table = pd.DataFrame(
            data={'image': filenames, 'label': predictions}
        )
        prediction_table.to_csv('neural_net.csv', header=False, index=False)
        return prediction_table



if __name__ == '__main__':

    train_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Train')
    valid_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Valid')
    test_dataset = DogHearUnlabeledDataset(data_root='Test')

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    net = NeuralNet(n_hiddens=64, n_classes=3, feature_extractor=SIFT(n_features=30))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        for images, labels, filenames in train_dataloader:
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}')
        
    test_dataset = DogHearUnlabeledDataset(data_root='Test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    net.predict(test_dataloader=test_dataloader)



