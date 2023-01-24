import os.path
import random

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from splitdata2 import SplitSegData
from transforms import *
from typing import Any, Callable, Optional, Tuple, TypeVar, Union
from pathlib import Path
from imageio.v2 import imread
from torch import Tensor


class PMSE_Dataset(Dataset):

    def __init__(self, image_dir: str,
                 label_dir: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 horizontal_split: Optional[Tuple[int, bool]] = None,
                 vertical_split: Optional[Tuple[int, bool]] = None,
                 square_split: bool = False,
                 percent_overlap: float = None,
                 last_min_overlap: Union[int, float] = None,
                 **kwargs):

        self.transform = transform
        self.target_transform = target_transform

        self.dataset = SplitSegData(image_dir=image_dir,
                                    label_dir=label_dir,
                                    horizontal_split=horizontal_split,
                                    vertical_split=vertical_split,
                                    square_split=square_split,
                                    split_overlap=percent_overlap,
                                    last_min_overlap=last_min_overlap,
                                    **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, mask, name = self.dataset[item]
        image = np.array([image])
        mask = np.array([mask])

        print(image.shape)
        image = image.transpose((1, 2, 0))
        mask = mask.transpose((1, 2, 0))
        print(image.shape)
        if self.transform:
            if isinstance(self.transform, PairCompose):
                image, mask = self.transform(image, mask)
            else:
                image = self.transform(image)
                if self.target_transform:
                    mask = self.transform(mask)

        return image, mask, name

    def get_original_image_and_mask(self, item):
        image, mask, name = self.dataset[item]
        image = np.array(image)
        mask = np.array(mask)

        return image, mask, name


class PMSEDatasetPresaved(Dataset):

    def __init__(self, data_path: str,
                 label_path: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None
                 ):
        super(PMSEDatasetPresaved, self).__init__()
        self.data_path = data_path
        self.label_path = label_path

        self.data = os.listdir(self.data_path)
        self.labels = os.listdir(self.label_path)

        self.transform = transform
        self.target_transform = target_transform

        for data in self.data:
            if not data.endswith('.png'):
                raise TypeError(f'Data and labels must be PNG images. Got {data} instead.')

        if not self.data == self.labels:
            raise ValueError('Image data and label data is not matching. Must be image and a label pairs with matching'
                             'names.')

    def __getitem__(self, item) -> Tuple[Tensor, Tensor]:
        data_name = self.data[item]
        label_name = self.labels[item]

        if data_name != label_name:
            raise ValueError(f'Data and label does not match. The data and label image must have same name. Got'
                             f'{data_name} and {label_name}.')

        image = imread(os.path.join(self.data_path, data_name))
        label = imread(os.path.join(self.label_path, label_name))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        label = torch.from_numpy(label.transpose((2, 0, 1)))

        if image.shape != label.shape:
            raise ValueError(f'Image and Mask must have same shape. Got {image.shape} and {label.shape}')

        h, w = image.shape[-2:]

        if h < w:
            rnd_num = random.randrange(0, w - h)
            image = image[:, :, rnd_num:rnd_num + h]
            label = label[:, :, rnd_num:rnd_num + h]

        return image, label

    def __len__(self):
        return len(self.data)


def get_dataloader(config_path: str, transforms, mode: str = 'train') -> DataLoader:
    valid_modes = ['train', 'validate', 'test']

    if mode not in valid_modes:
        raise ValueError(f'Mode {mode} is not a valid mode. Choose between {valid_modes}')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if mode == valid_modes[0]:
        data_path = config['path_train']
        pmse_data = os.path.join(data_path, 'data')
        pmse_label = os.path.join(data_path, 'label')

        dataset = PMSEDatasetPresaved(pmse_data,
                                      pmse_label,
                                      transform=transforms)

    elif mode == valid_modes[1]:
        data_path = config['path_validate']
        pmse_data = os.path.join(data_path, 'data')
        pmse_label = os.path.join(data_path, 'label')

        dataset = PMSE_Dataset(pmse_data,
                               pmse_label,
                               transform=transforms,
                               square_split=True,
                               percent_overlap=.0)

    elif mode == valid_modes[2]:
        data_path = config['path_test']
        pmse_data = os.path.join(data_path, 'data')
        pmse_label = os.path.join(data_path, 'label')

    else:
        raise Exception

    if mode == valid_modes[0]:
        dataset = PMSEDatasetPresaved(pmse_data,
                                      pmse_label,
                                      transform=transforms)

        dataloader = DataLoader(dataset,
                                batch_size=config['dataloader']['batch_size'],
                                num_workers=config['dataloader']['num_workers'],
                                shuffle=config['dataloader']['shuffle'])
    else:
        dataset = PMSE_Dataset(pmse_data,
                               pmse_label,
                               transform=transforms,
                               square_split=True,
                               percent_overlap=.0)

        dataloader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=config['dataloader']['num_workers'],
                                shuffle=False)

    return dataloader


if __name__ == '__main__':
    data_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Complete\\data'
    label_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Complete\\label'

    dataset_matload = PMSE_Dataset(data_path, label_path, transform=ToTensor())

    print(dataset_matload[0][0].shape)
    #dataset = PMSEDatasetPresaved(data_path, label_path)
    #i = dataset[0]
