from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import torch
from torch.utils.data import Dataset, DataLoader

from feature_extractors import FeatureExtractor, HOG, SIFT, SURF


class Predictor:

    def __init__(
        self, 
        model: BaseEstimator,
        feature_extractor: FeatureExtractor,
    ):
        self.model = model
        self.feature_extractor: FeatureExtractor = feature_extractor

    def fit(self, train_dataset: Dataset) -> None:
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
        
        images: torch.Tensor; labels: torch.Tensor; filenames: List[str]
        images, labels, filenames = next(iter(train_dataloader))
        
        images: np.ndarray = images.numpy()
        labels: np.ndarray = labels.numpy()
        
        train_features: List[np.ndarray] = []
        for image in images:
            feature = self.feature_extractor(image_array=image)
            train_features.append(feature)

        train_features = np.stack(arrays=train_features, axis=0)
        self.model.fit(X=train_features, y=labels)

    def predict(self, test_dataset: Dataset) -> np.ndarray:
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
        
        image: torch.Tensor; filenames: List[str]
        images, filenames = next(iter(test_dataloader))

        images: np.ndarray = images.numpy()

        test_features: List[np.ndarray] = []
        for image in images:
            feature = self.feature_extractor(image_array=image)
            test_features.append(feature)

        predicted_labels: np.ndarray = self.model.predict(X=test_features)
        prediction_table = pd.DataFrame(
            data={'image': filenames, 'label': predicted_labels}
        )
        prediction_table.to_csv(
            f'{self.model.__class__.__name__}_{self.feature_extractor.__class__.__name__}.csv', 
            header=False, 
            index=False,
        )
        return prediction_table

if __name__ == '__main__':
    from datasets import DogHeartLabeledDataset, DogHearUnlabeledDataset
    predictor = Predictor(model=SVC())
    train_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Train')
    test_dataset = DogHearUnlabeledDataset('Test')
    predictor.fit(train_dataset=train_dataset)
    a = predictor.predict(test_dataset=test_dataset)

