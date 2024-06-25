from typing import List
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import DogHeartLabeledDataset, DogHearUnlabeledDataset
from feature_extractors import HOG, SIFT
from models import Predictor


train_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Train')
test_dataset = DogHearUnlabeledDataset('Test')

# hog_svc = Predictor(model=SVC(), feature_extractor=HOG(channel_axis=2))
# hog_svc.fit(train_dataset=train_dataset)
# hog_svc.predict(test_dataset=test_dataset)


sift_knn = Predictor(model=KNeighborsClassifier(), feature_extractor=SIFT(n_features=30))
sift_knn.fit(train_dataset=train_dataset)
sift_knn.predict(test_dataset=test_dataset)


