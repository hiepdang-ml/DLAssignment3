from abc import ABC, abstractmethod
from typing import List

from datasets import DogHeartLabeledDataset

import numpy as np
import cv2
import skimage.feature

import torch
import torch.utils


class FeatureExtractor(ABC):

    @abstractmethod
    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        pass


class HOG(FeatureExtractor):

    def __init__(self, channel_axis: int = 0) -> None:
        self.channel_axis: int = channel_axis

    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        return skimage.feature.hog(image=image_array)
    

class SIFT:

    def __init__(self, n_features: int):
        self.n_features: int = n_features
        self.sift: cv2.SIFT = cv2.SIFT_create()

    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        keypoints: List[cv2.KeyPoint]
        descriptors: np.ndarray
        keypoints, descriptors = self.sift.detectAndCompute(
            image=(image_array * 255).astype(np.uint8), 
            mask=None,
        )   # (keypoints, descriptors)

        assert len(keypoints) == descriptors.shape[0]
        indices: np.ndarray = np.argsort([kp.response for kp in keypoints])[::-1][:self.n_features]

        keypoints: List[cv2.KeyPoint] = [keypoints[idx] for idx in indices]
        features: np.ndarray = descriptors[indices]
        if features.shape[0] < self.n_features:
            padding = np.zeros(shape=(self.n_features - features.shape[0], features.shape[1]))
            features = np.concatenate((features, padding), axis=0)
        
        features: np.ndarray = features.flatten()
        return features


class SURF:

    def __init__(self, n_features: int):
        self.n_features: int = n_features
        self.surf: cv2.xfeatures2d_SURF = cv2.xfeatures2d.SURF_create(nOctaves=5)

    def __call__(self, image_array: np.ndarray) -> np.ndarray:
        keypoints: List[cv2.KeyPoint]
        descriptors: np.ndarray
        keypoints, descriptors = self.surf.detectAndCompute(
            image=(image_array * 255).astype(np.uint8), 
            mask=None,
        )   # (keypoints, descriptors)

        assert len(keypoints) == descriptors.shape[0]
        indices: np.ndarray = np.argsort([kp.response for kp in keypoints])[::-1][:self.n_features]

        keypoints: List[cv2.KeyPoint] = [keypoints[idx] for idx in indices]
        features: np.ndarray = descriptors[indices]
        if features.shape[0] < self.n_features:
            padding = np.zeros(shape=(self.n_features - features.shape[0], features.shape[1]))
            features = np.concatenate((features, padding), axis=0)

        features = features.flatten()
        return features


if __name__ == '__main__':

    train_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Train')
    valid_dataset = DogHeartLabeledDataset(data_root='Dog_heart/Valid')

    # P1.Q2
    hog = HOG(channel_axis=2)

    train_hog_features: List[np.ndarray] = []
    for image, label, filename in train_dataset:
        hog_feature = hog(image_array=image.numpy())
        train_hog_features.append(hog_feature)

    train_hog_features = np.stack(arrays=train_hog_features, axis=0)

    # valid_hog_features: List[np.ndarray] = []
    # for image, label, filename in valid_dataset:
    #     hog_feature, _ = hog(image_array=image.numpy())
    #     valid_hog_features.append(hog_feature)

    # valid_hog_features = np.stack(arrays=valid_hog_features, axis=0)

    # print(train_hog_features.shape)
    # print(valid_hog_features.shape)


    # # P1.Q3
    # sift = SIFT(n_features=30)

    # train_sift_features: List[np.ndarray] = []
    # for image, label, filename in train_dataset:
    #     sift_feature = sift(image_array=image.numpy())
    #     train_sift_features.append(sift_feature)

    # train_sift_features = np.stack(arrays=train_sift_features, axis=0)

    # P1.Q4
    surf = SURF(n_features=30)

    train_surf_features: List[np.ndarray] = []
    for image, label, filename in train_dataset:
        surf_feature = surf(image_array=image.numpy())
        train_surf_features.append(surf_feature)

    train_surf_features = np.stack(arrays=train_surf_features, axis=0)

    # P1.Q5






