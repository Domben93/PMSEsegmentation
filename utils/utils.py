import copy
import datetime
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.io import loadmat
from scipy.interpolate import NearestNDInterpolator
from typing import Union, List, Tuple, Dict, Type, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets.folder import IMG_EXTENSIONS
from collections import OrderedDict
from pathlib import Path
from imageio.v2 import imread
from dotmap import DotMap
import yaml


__all__ = [
    'default_loader',
    'mat_loader',
    'pil_loader',
    'dataset_mean_and_std',
    'CustomOrderedDict',
    'min_max_dataset',
    'save_model',
    'load_yaml_as_dotmap'
]


def default_loader(path: str, return_type: Union[Type[Image.Image], Type[np.ndarray]] = np.ndarray, **kwargs) -> Union[
                   Image.Image, np.ndarray]:

    from pathlib import Path

    if not os.path.exists(path):
        raise FileNotFoundError(f'Path: {path} does not exist!')

    if Path(path).suffix == '.mat':
        data = mat_loader(path, **kwargs)

    elif Path(path).suffix == '.png':
        data = imread(path)

    else:
        data = pil_loader(path, **kwargs)

    if isinstance(data, return_type):
        return data
    else:
        if isinstance(data, Image.Image):
            return np.array(data)
        else:
            return Image.fromarray(data)


def mat_loader(path: str, **kwargs) -> np.ndarray:
    """
    Loads array from .mat file
    Args:
        path: path to directory:
        **kwargs:
            array_name: name of array in dict
            array_position: Position of array in dict
            image_mode: PIL.Image mode (e.g 'L': 8-bit pixels, black and white)

    Returns: PIL.Image.Image

    """

    dict_name = kwargs.get('array_name')
    dict_position = kwargs.get('array_position')
    data = None
    data_dict = list(loadmat(path).keys())

    if dict_name in data_dict:
        data = loadmat(path).get(dict_name)
    elif dict_position:
        data = loadmat(path)[data_dict[dict_position]]
    else:
        for item in data_dict:
            data = loadmat(path)[item]
            if isinstance(data, np.ndarray):
                break

    if not isinstance(data, np.ndarray):
        raise TypeError(f'Need {repr(np.ndarray)} but got <{repr(type(data))}> instead.')

    if np.isnan(data).any():
        if data.shape[0] == 1:
            raise ValueError(f'Filtering out NaN values is not supported for 1D arrays. Got array shape ({data.shape})')
        idx = np.where(~np.isnan(data))

        interpolate_nans = NearestNDInterpolator(np.transpose(idx), data[idx])
        data = interpolate_nans(*np.indices(data.shape))

    return data


def pil_loader(path: str, **kwargs) -> Image.Image:
    """
    loads images
    Args:
        path:
        **kwargs:
            image_mode: PIL.Image mode (e.g 'L': 8-bit pixels, black and white)

    Returns:

    """

    if kwargs.get('image_mode') is None:
        mode = 'L'
    else:
        mode = kwargs.get('image_mode')

    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode=mode)


def dataset_mean_and_std(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    count = 0

    mean = torch.empty(1)
    std = torch.empty(1)

    for image, _, _ in tqdm(dataloader, desc='Finding mean and std of dataset'):
        b, c, h, w = image.shape
        num_pixels = b * h * w
        if torch.isnan(image).any():
            raise ValueError(f'NaN value detected! This will corrupt mean and std results')
        sum_ = torch.sum(image, dim=[0, 2, 3])
        sum_squ = torch.sum(torch.square(image), dim=[0, 2, 3])

        mean = (count * mean + sum_) / (count + num_pixels)
        std = (count * std + sum_squ) / (count + num_pixels)

        count += num_pixels

    std = torch.sqrt(std - torch.square(mean))

    return mean, std


def min_max_dataset(dataloader: DataLoader) -> Tuple[float, float]:
    """
    Finding min and max value over the entire dataset. Remember that any transformations given to the dataset class
    might affect the result. Instead, it is recommended to use the raw dataset
    Args:
        dataloader: Dataloader class instance

    Returns:
        minimum and maximum values of the dataset given by the dataloader.
    """
    min_ = torch.tensor(float('inf')).max()
    max_ = torch.tensor(float('-inf')).max()

    for image, _, _ in dataloader:
        if torch.isnan(image).any():
            raise ValueError(f'NaN value detected! This will corrupt mean and std results')
        image = image[0]
        im_min = torch.min(image)
        if im_min < min_:
            min_ = im_min

        im_max = torch.max(image)
        if im_max > max_:
            max_ = im_max
            print(max_)

    return min_, max_


class CustomOrderedDict(OrderedDict):
    from typing import TypeVar
    _T = TypeVar('_T')

    def __init__(self, val=None):
        if val is None:
            val = {}
        super().__init__(val)

    @classmethod
    def fromkeys(cls, iterable: list, value: Optional[Union[List, Dict]] = None) -> dict[Any, Any]:

        self = cls()
        if value is None:
            for key in iterable:
                self[key] = value
        elif isinstance(value, list):
            if len(iterable) != len(value):
                raise ValueError(f'List "iterable" and "value" must have same length. {len(iterable)} != {len(value)}')
            else:
                for key, type_ in zip(iterable, value):
                    if type_ is None:
                        self[key] = None
                    else:
                        self[key] = type_()

        elif isinstance(value, tuple(CustomOrderedDict.mro())):
            for key in iterable:
                self[key] = copy.deepcopy(value)
        else:
            raise ValueError('Must be a list or NoneType')

        return self


MODEL_EXTENSION = ['.pt', '.pth']


def save_model(model: nn.Module,
               optimizer: Any,
               epoch: int,
               train_samples: int,
               file_name: str,
               save_path: str = '',
               loss_history: Dict = None,
               model_state_dict: Optional[Dict] = None,
               optimizer_state_dict: Optional[Dict] = None
               ):

    if not model_state_dict and optimizer_state_dict:
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

    elif (model_state_dict or optimizer_state_dict) is None:
        raise ValueError(f'Model state dict and optimizer state dict must be given as pair or not at all!')

    state = {'sys_argv': sys.argv,
             'time': str(datetime.datetime.now()),
             'model_name': type(model).__name__,
             'model_state': model_state_dict,
             'optimizer_name': type(optimizer).__name__,
             'optimizer_state': optimizer_state_dict,
             'epoch': epoch,
             'totalTrainingSamples': train_samples,
             'loss': loss_history
             }

    if not Path(file_name).suffix in MODEL_EXTENSION:
        raise ValueError(f'File name must have one of the valid extensions: {MODEL_EXTENSION}. '
                         f'Got {Path(file_name).suffix}')

    if os.path.exists(save_path):
        full_path = os.path.join(save_path, file_name)
    else:
        warnings.warn(f'Could not find specified path: {save_path}. Saving at {os.getcwd()}')
        full_path = os.path.join(os.getcwd(), file_name)

    torch.save(state, full_path)


def load_yaml_as_dotmap(yaml_path: str) -> DotMap:

    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    config = DotMap(config, _dynamic=True)

    return config

"""
if __name__ == '__main__':
    list_ = ['1', '2', '3', '4']
    image_info_list = ['RowSplit', 'ColumnSplit', 'ImagesAfterSplit', 'RowOverlap', 'ColOverlap']
    type_list = [list, list, None, None, None]
    split_info = CustomOrderedDict.fromkeys(list_, CustomOrderedDict.fromkeys(image_info_list, type_list))

    split_info['1']['RowSplit'].append(1)
    print(split_info)

    print(split_info['1']['RowSplit'] is split_info['2']['RowSplit'])
"""
