import copy
import datetime
import os
import sys
import warnings
import yaml
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
from scipy.io import loadmat
from scipy.interpolate import NearestNDInterpolator
from typing import Union, List, Tuple, Dict, Type, Any, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from collections import OrderedDict
from pathlib import Path
from imageio.v2 import imread
from dotmap import DotMap


__all__ = [
    'default_loader',
    'mat_loader',
    'pil_loader',
    'dataset_mean_and_std',
    'CustomOrderedDict',
    'min_max_dataset',
    'save_model',
    'load_yaml_as_dotmap',
    'remove_from_dataname',
    'EarlyStopper',
    'remove_from_dataname_extended',
    'save_images',
    'display',
    'check_extension',
    'save_results'
]

EXTENSIONS = ['.mat', '.png', '.jpg']


def check_extension(path) -> Tuple[bool, str]:
    extension = None
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):
            if extension is None:
                extension = os.path.splitext(filename)[1]
            elif os.path.splitext(filename)[1] != extension:
                return False, extension
            elif os.path.splitext(filename)[1] not in EXTENSIONS:
                return False, extension

    return True, extension


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

    if len(data.shape) == 3:
        pass
    elif len(data.shape) == 2:
        data = np.array([data]).transpose((1, 2, 0))
    else:
        raise ValueError(f'Data has to be either of size [N, H, W] or [H, W]. [H, W] shape is automatically'
                         f' reshaped to [1, H, W]. Got {len(data.shape)}.')

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


def remove_from_dataname(names: List[str]):
    new_list = []

    for name in names:
        new_list.append(name.rpartition('_')[0].rpartition('_')[0])

    return new_list


def remove_from_dataname_extended(names: List[str], samples: List[List[str]]):
    new_list = []
    samples = [string for sublist in samples for string in sublist]

    for name in names:
        index = [i for i, s in enumerate(samples) if s in name]
        if len(index) > 1:
            raise ValueError('Found more than one positions where the same name is contained')

        new_list.append(samples[index[0]])

    return new_list


def load_yaml_as_dotmap(yaml_path: str) -> DotMap:
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    config = DotMap(config, _dynamic=True)

    return config


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, increasing=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def display(images, labels, preds, info, sample_classes):
    if not len(images) == len(labels) == len(preds):
        raise ValueError('List of images, labels and predictions must be of equal length')

    for sample_class in sample_classes:

        fig, ax = plt.subplots(4, len(sample_class))
        fig.add_gridspec(4, len(sample_class), wspace=0.0, hspace=0.0)

        for n, name in enumerate(sample_class):

            name_indices = [i for i, j in enumerate(info) if j == name]

            if len(name_indices) > 1:
                im = torch.cat([images[index] for index in name_indices], dim=1)
                label = torch.cat([labels[index] for index in name_indices], dim=1)
                pred = torch.cat([preds[index] for index in name_indices], dim=1)
            else:
                im = images[name_indices[0]]
                label = labels[name_indices[0]]
                pred = preds[name_indices[0]]

            ax[0, n].imshow(im[:, :], cmap='jet', vmin=0, vmax=1)
            ax[1, n].imshow(label[:, :], cmap='jet', vmin=0, vmax=1)
            ax[2, n].imshow(pred[:, :], cmap='jet', vmin=0, vmax=1)
            ax[3, n].imshow(np.where(pred[:, :] >= 0.5, 1, 0), cmap='jet', vmin=0, vmax=1)
            ax[0, n].axis('off')
            ax[1, n].axis('off')
            ax[2, n].axis('off')
            ax[3, n].axis('off')
        plt.show()


def save_images(images, labels, preds, save_path, names):

    if not len(images) == len(labels) == len(preds):
        raise ValueError('List of images, labels and predictions must be of equal length')

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'images'))
        os.mkdir(os.path.join(save_path, 'labels'))
        os.mkdir(os.path.join(save_path, 'preds'))

    for info in names:

        name_indices = [i for i, j in enumerate(names) if j == info]

        if len(name_indices) > 1:
            im = torch.cat([images[index] for index in name_indices], dim=1)
            label = torch.cat([labels[index] for index in name_indices], dim=1)
            pred = torch.cat([preds[index] for index in name_indices], dim=1)
        else:
            im = images[name_indices[0]]
            label = labels[name_indices[0]]
            pred = preds[name_indices[0]]

        save_image(im, save_path + '/images/' + f'im_{info}' + '.png')
        save_image(label, save_path + '/labels/' + f'mask_{info}' + '.png')
        save_image(pred, save_path + '/preds/' + f'label_{info}' + '.png')


def save_results(preds, save_path, names):

    for info in names:

        name_indices = [i for i, j in enumerate(names) if j == info]

        if len(name_indices) > 1:
            pred = torch.cat([preds[index] for index in name_indices], dim=1)
        else:
            pred = preds[name_indices[0]]

        save_image(pred, save_path + f'result_{info}' + '.png')


if __name__ == '__main__':
    sample_list_ = [['MAD6400_2008-07-02_arcd_60@vhf_400749', 'MAD6400_2009-06-10_arcd_60@vhf_422844'],
                    ['MAD6400_2008-06-30_manda_60@vhf_060693', 'MAD6400_2009-07-14_manda_60@vhf_684778',
                     'MAD6400_2009-07-16_manda_60@vhf_655441', 'MAD6400_2009-07-17_manda_60@vhf_633279',
                     'MAD6400_2009-07-30_manda_60@vhf_779294', 'MAD6400_2010-07-07_manda_60@vhf_576698',
                     'MAD6400_2010-07-08_manda_59', 'MAD6400_2010-07-09_manda_60@vhf_470083'],
                    ['MAD6400_2015-08-10_manda_59', 'MAD6400_2015-08-12_manda_59',
                     'MAD6400_2015-08-13_manda_59', 'MAD6400_2015-08-20_manda_59'],
                    ['MAD6400_2011-06-01_manda_59', 'MAD6400_2011-06-08_manda_59',
                     'MAD6400_2011-06-09_manda_59', 'MAD6400_2014-07-01_manda_48@vhf_178333']]

    info_list = ['MAD6400_2008-07-02_arcd_60@vhf_400749_1', 'MAD6400_2009-06-10_arcd_60@vhf_422844_1',
                 'MAD6400_2014-07-01_manda_48@vhf_178333_1']

    info_list = remove_from_dataname_extended(info_list, sample_list_)
    print(info_list)

    new_sample_list = [[subelt for subelt in elt if subelt in info_list] for elt in sample_list_]
    new_sample_list = [x for x in new_sample_list if x != []]
    print(new_sample_list)