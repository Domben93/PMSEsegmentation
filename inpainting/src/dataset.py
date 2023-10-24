import math
from glob import glob
from typing import List, Optional, Sequence

import torch
from PIL import Image
import random

from torch import Tensor
from torch.utils.data import Dataset
import os
from torchvision.transforms import functional as f
from torch.nn.functional import pad
from imageio.v2 import imread
import math


class QuasiResize:
    """
    Resize the image such that the original image is scaled by integers and then pads the area left
    by a chosen padding-mode such that it fits the wanted output size. In case of odd number of padding instances
    the top and right will be padded with one extra value in order to achieve the wanted output size
    """

    def __init__(self, size: List[int], max_scaling: int, padding_mode: str = None, value: int = 0,
                 interpolation: Optional[f.InterpolationMode] = f.InterpolationMode.NEAREST):
        """

        Args:
            size:
            max_scaling:
            padding_mode:
            value:
            interpolation:
        """

        if not isinstance(size, (int, Sequence)):
            raise TypeError(f'Size must be int or sequence of int. Got {type(size)} instead')
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError(f'If size is a sequence, one or two values are required. Got {size}')

        self.size = size
        self.padding = padding_mode
        self.scale = max_scaling
        self.interpolation = interpolation
        if self.padding is None:
            self.padding = 'constant'

        if self.padding == 'constant':
            self.value = value
        else:
            self.value = 0

    def __call__(self, image: Tensor) -> Tensor:

        if not isinstance(image, Tensor):
            raise TypeError(f'Image must be a torch Tensor. Got {type(image)} instead.')
        if not len(image.shape) >= 3:
            raise ValueError(f'Tensor is expected to have [..., H, W] shape. Got {image.shape} instead.')
        h, w = image.shape[-2], image.shape[-1]

        if self.size[0] == self.size[1]:
            im_size = self.size[0]
            h_scale = im_size // h if im_size // h <= self.scale else self.scale
            w_scale = im_size // w if im_size // w <= self.scale else self.scale

            image = f.resize(image, [h * h_scale, w * w_scale], self.interpolation)

        else:
            h_size, w_size = self.size[0], self.size[1]
            h_scale = h_size // h if h_size // h else self.scale
            w_scale = w_size // w if w_size // w else self.scale

            image = f.resize(image, [h * h_scale, w * w_scale], self.interpolation)

        h_pad_val = self.size[0] - image.shape[-2]
        w_pad_val = self.size[1] - image.shape[-1]

        padding = [0] * 4

        if h_pad_val:
            padding[2] = math.ceil(h_pad_val / 2)  # left padding value
            padding[3] = math.floor(h_pad_val / 2)  # right padding value

        if w_pad_val:
            padding[0] = math.ceil(w_pad_val / 2)  # top padding value
            padding[1] = math.floor(w_pad_val / 2)  # bottom padding value

        image = pad(image, pad=padding, mode=self.padding, value=self.value)

        return image


class Places2(Dataset):
    def __init__(self, data_root, mask_root, data='train'):
        super(Places2, self).__init__()
        if data == 'train':
            self.data_root = os.path.join(data_root, 'train')
        else:
            self.data_root = os.path.join(data_root, 'validation')
        # get the list of image paths

        self.images = os.listdir(os.path.join(self.data_root, 'data'))
        self.masks = os.listdir(os.path.join(mask_root, 'generated_masks'))

        self.N_mask = len(self.masks)
        self.resize = QuasiResize([64, 64], 2, padding_mode='zeros')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img = imread(os.path.join(self.data_root, 'data', self.images[index]))
        img = f.to_tensor(img)

        msk_indices = [i for i, s in enumerate(self.masks) if
                       self.images[index].removesuffix('.png') in s.removesuffix('.png')]

        mask = imread(os.path.join(self.data_root, 'generated_masks', self.masks[random.choice(msk_indices)]))
        mask = f.to_tensor(mask)

        c, h, w = img.shape

        if w - h < 0:
            r_int = 0
        else:
            r_int = random.randint(0, w - h)

        #img = self.resize(img[:1, :, r_int:r_int + h])
        #mask = self.resize(mask[:1, :, r_int:r_int + h])
        img = f.resize(img[:1, :, r_int:r_int + h], [64, 64], interpolation=f.InterpolationMode.NEAREST)
        mask = f.resize(mask[:1, :, r_int:r_int + h], [64, 64], interpolation=f.InterpolationMode.NEAREST)

        return img * mask, mask, img


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    root = 'C:\\Users\\dombe\\PycharmProjects\\Test\\partial_conv\\data'

    dataset = Places2(root)

    msk_img, msk, img = dataset[2]

    fig, ax = plt.subplots(2, 1)

    ax[0].imshow(msk[0, :, :], cmap='jet', vmin=0, vmax=1)
    ax[1].imshow(img[0, :, :], cmap='jet', vmin=0, vmax=1)
    #ax[2].imshow(img[2, 0, :, :], cmap='jet', vmin=0, vmax=1)
    #ax[3].imshow(img[3, 0, :, :], cmap='jet', vmin=0, vmax=1)
    plt.show()
