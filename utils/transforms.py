import random

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Generic, TypeVar, List, Tuple, Optional, Any, NoReturn, Literal, Sequence, Callable
import math
from PIL import Image
import torch
import torch.nn as nn
from torch import Tensor
from numpy import ndarray
from torchvision.transforms import functional as f
from torch.nn.functional import pad
from scipy.interpolate import griddata
from skimage.restoration import inpaint
from skimage import measure
from partial_conv.src.model import PConvUNet

__all__ = [
    "Compose",
    "ToTensor",
    "ToGrayscale",
    "ToPIL",
    "Standardize",
    "QuasiResize",
    "UndoQuasiResize",
    "Normalize",
    "PairCompose",
    "MaskClassReduction",
    "Transpose",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "RandomResize",
    "RandomRotation",
    "Crop",
    'FunctionalTransform',
    'RandomContrastAdjust',
    'RandomBrightnessAdjust',
    'RandomPositionShift'
]


class Transform(object):

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class FunctionalTransform(object):

    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


TransformTypes = TypeVar("TransformTypes", Transform, FunctionalTransform, type(None))


class Compose:

    def __init__(self, transforms: List[TransformTypes]):

        if not all(isinstance(transform, Transform) for transform in transforms):
            raise TypeError(f'All elements in transforms list must be a object of Class Transform')
        self.transforms = transforms

    def __call__(self, image: Any) -> Any:
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self) -> str:
        list_string = self.__class__.__name__ + '<'
        for t in self.transforms:
            list_string += '\n'
            list_string += f'{t}'
        list_string += '\n>'
        return list_string


class PairCompose:

    def __init__(self, transforms: List[List[TransformTypes]]):

        if not all(isinstance(transform, (FunctionalTransform, Transform, type(None)))
                   for transform in [item for sublist in transforms for item in sublist]):
            raise TypeError(f'List can only contain Transforms, FunctionalTransforms or NoneType types.')

        self.transforms = transforms

    def __call__(self, image: Any, mask: Any) -> Tuple[Any, Any]:

        for num, transform_pair in enumerate(self.transforms):
            if 0 == len(transform_pair) > 2:
                raise ValueError(
                    f'For each transform pair it is only allowed with one or two transforms'
                    f'(or NoneType if no transform is wanted for either target or image).\n'
                    f'Got {len(transform_pair)} at index {num} with the transforms: {transform_pair}')

            if all(isinstance(transform, (Transform, type(None))) for transform in transform_pair) \
                    and len(transform_pair) == 2:

                if transform_pair[0]:
                    image = transform_pair[0](image)
                if transform_pair[1]:
                    mask = transform_pair[1](mask)

            elif isinstance(transform_pair[0], FunctionalTransform) and len(transform_pair) == 1:

                image, mask = transform_pair[0](image, mask)

            else:
                raise Exception(f'')

        return image, mask

    def __repr__(self) -> str:
        list_string = self.__class__.__name__ + '<'
        for t in self.transforms:
            list_string += '\n'
            for i in t:
                list_string += f'{i}'
        list_string += '\n>'
        return list_string


class ConvertDtype(Transform):

    def __init__(self, dtype: torch.dtype):
        super(ConvertDtype, self).__init__()
        self.dtype = dtype

    def __call__(self, image: Tensor) -> Tensor:
        return image.to(self.dtype)


class ToTensor(Transform):
    def __init__(self, zero_one_range: bool = True):
        super(ToTensor, self).__init__()
        self.range = zero_one_range

    def __call__(self, image: Union[Image.Image, ndarray]) -> Tensor:
        if self.range:
            return f.to_tensor(image)
        else:
            return torch.tensor(image, dtype=torch.float64)

    def __repr__(self) -> str:
        return self.__class__.__name__


class ToNdarray(Transform):
    # torch.Tensor or PIL.Image.Image to numpy.ndarray
    def __call__(self, image: Union[Image.Image, Tensor]) -> ndarray:
        if isinstance(image, Tensor):
            image = image.cpu().np().transpose((1, 2, 0))
            if image != np.uint8:
                image = image.astype(np.uint8)
            return image
        elif isinstance(image, Image.Image):
            return np.array(image, dtype=np.uint8)
        else:
            raise Exception(f'{self.__class__.__name__} does not support type {repr(type(image))}')

    def __repr__(self) -> str:
        return self.__class__.__name__


class ToPIL(Transform):
    # torch.Tensor or numpy.ndarray to PIL.Image.Image
    def __call__(self, image: Union[ndarray, Tensor]) -> Image.Image:
        return f.to_pil_image(image)

    def __repr__(self) -> str:
        return self.__class__.__name__


class ToGrayscale(Transform):
    def __init__(self, output_channels: int = 1):
        super(ToGrayscale, self).__init__()
        self.output_channels = output_channels

    def __call__(self, image: Union[Tensor, ndarray]) -> Union[Tensor]:
        if isinstance(image, ndarray):
            image = torch.from_numpy(image)
        if len(image.shape) < 2:
            raise ValueError('Image must have at least 3 dimensions with channels, height and width; [..., C, H, W]')
        if len(image.shape) == 2:
            image = image.view((1, int(image.shape[0]), int(image.shape[1])))
        if image.shape[-3] not in (1, 3):
            raise ValueError(f'Number of channels must be either 1 or 3. Got {image.shape[-3]}')

        if image.shape[-3] > 1:
            image = f.rgb_to_grayscale(image, num_output_channels=self.output_channels)
        else:
            image = image.repeat(3, 1, 1)  # stack same values in channel dimensions
        return image


class QuasiResize(Transform):
    """
    Resize the image such that the original image is scaled by integers and then pads the area left
    by a chosen padding-mode such that it fits the wanted output size. In case of odd number of padding instances
    the top and right will be padded with one extra value in order to achieve the wanted output size
    """
    PADDING_MODE = Literal['constant', 'reflect', 'replicate', 'circular', None]

    def __init__(self, size: List[int], max_scaling: int, padding_mode: PADDING_MODE = None, value: int = 0,
                 interpolation: Optional[f.InterpolationMode] = f.InterpolationMode.NEAREST):
        """

        Args:
            size:
            max_scaling:
            padding_mode:
            value:
            interpolation:
        """
        super(QuasiResize, self).__init__()

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


class UndoQuasiResize:

    def __init__(self, quasiresize: QuasiResize):
        super(UndoQuasiResize, self).__init__()

        self.size = quasiresize.size
        self.padding = quasiresize.padding
        self.scale = quasiresize.scale
        self.interpolation = quasiresize.interpolation

    def __call__(self, image: Tensor, original_size: Tuple[Any, Any]) -> Tensor:

        h, w = image.shape[-2], image.shape[-1]
        o_h, o_w = original_size[-2], original_size[-1]

        if o_h * self.scale < h:
            h_multiplier = self.scale
        else:
            h_multiplier = 1

        if o_w * self.scale < w:
            w_multiplier = self.scale
        else:
            w_multiplier = 1

        left = math.ceil((w - (o_w * w_multiplier)) / 2)
        top = math.ceil((h - (o_h * h_multiplier)) / 2)

        image = f.crop(image, top=top,
                       left=left,
                       height=o_h * h_multiplier,
                       width=o_w * w_multiplier)

        if len(image.shape) == 2:
            image = image.view(1, image.shape[-2], image.shape[-1])

        image = f.resize(image, [o_h, o_w], self.interpolation)

        """
        if h == w:
            actual_scale = 0
            for i in range(1, self.scale + 1):
                if o_h * (i + 1) >= h:
                    actual_scale = i
                    break

            image_padding = (h - actual_scale * o_h) / 2
            image = image[::, math.ceil(image_padding):-math.floor(image_padding),
                    math.ceil(image_padding):-math.floor(image_padding)]

        else:
            actual_h_scale = 0
            actual_w_scale = 0

            for i in range(1, self.scale + 1):
                if o_h * (i + 1) >= h:
                    actual_h_scale = i

                if o_w * (i + 1) >= w:
                    actual_w_scale = i

                if actual_h_scale != 0 and actual_w_scale != 0:
                    break

            image_w_padding = (w - actual_w_scale * o_w) / 2
            image_h_padding = (h - actual_h_scale * o_h) / 2

            image = image[::, math.ceil(image_h_padding):-math.floor(image_h_padding),
                    math.ceil(image_w_padding):-math.floor(image_w_padding)]
        
        image = f.resize(image, [o_h, o_w], self.interpolation)
        """
        return image


class Standardize(Transform):

    def __init__(self, mean: Union[List[float], float], std: [list[float], float], inplace: bool = False):
        super(Standardize, self).__init__()

        if isinstance(mean, list) and isinstance(std, list):
            if len(mean) != len(std):
                raise ValueError(f'Mean and Std list must have same length. Got {len(mean)} and {len(std)}')
        elif isinstance(mean, float) and isinstance(std, float):
            pass
        else:
            raise TypeError('Mean and Std be of same type and of either type List of Float')

        self.mean = mean
        self.std = std

        self.inplace = inplace

    def __call__(self, image: Tensor) -> Tensor:

        if image.shape[-3] == 3 and isinstance(self.mean, float):
            self.mean = self.mean * 3

        elif image.shape[-3] == 1 and isinstance(self.mean, list):
            if image.shape[-3] != len(self.mean):
                raise ValueError(
                    f'Number of channels in image does not match. Got image shape: {image.shape} and mean and std '
                    f'shape: {len(self.mean)} & {len(self.std)}')
            else:
                self.mean = [self.mean[0]]

        return f.normalize(image, mean=self.mean, std=self.std, inplace=self.inplace)


class Normalize(Transform):

    def __init__(self, min_max_range: Tuple[float, float],
                 dataset_min_max: Tuple[float, float],
                 return_type: torch.dtype = None):
        super(Normalize, self).__init__()
        self.min_range, self.max_range = min_max_range
        self.min = dataset_min_max[0]
        self.max = dataset_min_max[1]
        self.return_type = return_type

    def __call__(self, image: Tensor) -> Tensor:
        """
        image = image.numpy().squeeze()
        scaler = preprocessing.MinMaxScaler(feature_range=self.feature_range)
        scaler.fit(image)
        return scaler.transform(image)
        """
        if self.return_type:
            image.type(self.return_type)

        image = image.squeeze()

        X_std = (image - self.min) / (self.max - self.min)
        image = X_std * (self.max_range - self.min_range) + self.min_range

        return image


class MaskClassReduction(Transform):
    """
    Remove classes from mask.
    """

    def __init__(self,
                 all_classes: List[int],
                 keep_classes: List[int],
                 replace_val: Union[int, List[int]] = None):

        super(MaskClassReduction, self).__init__()

        self.classes = all_classes
        self.classes = list(np.unique(np.array(self.classes)))

        self.keep_classes = keep_classes
        self.replace_val = replace_val if replace_val is not None else 0
        if self.replace_val and isinstance(self.replace_val, list):
            assert len(self.keep_classes) == len(self.replace_val), f'The list size of classes to keep and values ' \
                                                                    f'to replace them with must be of equal length' \
                                                                    f'. Got {len(self.keep_classes)} and ' \
                                                                    f'{len(self.replace_val)} respectively.'
            assert len(self.classes) >= len(self.keep_classes), 'More classes to keep given than amount of' \
                                                                ' original classes given.'
        elif isinstance(self.replace_val, int):
            self.replace_val = [self.replace_val] * len(self.keep_classes)

        else:
            raise TypeError(f'Expected Int or list of Int. Got {type(self.replace_val)}')

        self.remove_list = [x for x in self.classes if x not in self.keep_classes]

    def __call__(self, mask: Tensor) -> Tensor:

        for class_, new_val in zip(self.remove_list, self.replace_val):
            if self.replace_val:
                mask = torch.where(mask == class_, new_val, mask)

        return mask


class Transpose(Transform):

    def __init__(self, new_axis: Tuple[int, ...]):
        super(Transpose, self).__init__()
        self.axis = new_axis

    def __call__(self, array: ndarray) -> ndarray:
        return array.transpose(self.axis)


class RandomHorizontalFlip(FunctionalTransform):

    def __init__(self, p: float = 0.5):
        super(RandomHorizontalFlip, self).__init__()
        if 0 >= p >= 1:
            raise ValueError(f'Probability must be between 0.0 and 1.0. Got {p}')
        self.p = p

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            image = f.hflip(image)
            mask = f.hflip(mask)

        return image, mask


class RandomVerticalFlip(FunctionalTransform):

    def __init__(self, p: float = 0.5):
        super(RandomVerticalFlip, self).__init__()
        if 0 >= p >= 1:
            raise ValueError(f'Probability must be between 0.0 and 1.0. Got {p}')
        self.p = p

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            image = f.vflip(image)
            mask = f.vflip(mask)

        return image, mask


class Crop(FunctionalTransform):

    def __init__(self):
        super(Crop, self).__init__()
        raise NotImplementedError("Class not implemented")

    def __call__(self, image, mask):
        pass


class RandomResize(FunctionalTransform):

    def __init__(self, p, scale: Sequence):
        super(RandomResize, self).__init__()
        if 0 >= p >= 1:
            raise ValueError(f'Probability must be between 0.0 and 1.0. Got {p}')
        self.p = p
        self.scale = scale

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:

        if torch.rand(1) < self.p:
            if len(self.scale) == 2:
                scale_h = random.randint(self.scale[0], self.scale[1])
                scale_w = random.randint(self.scale[0], self.scale[1])

            elif len(self.scale) > 2:

                scale_h = random.choice(self.scale)
                scale_w = random.choice(self.scale)

            else:
                raise TypeError(f'Expected Sequence of numbers that can be chosen from. '
                                f'Got {type(self.scale)}')


            h, w = int(image.shape[-2] + scale_h), int(image.shape[-1] + scale_w)

            if h <= 0:
                h = 1
            if w <= 0:
                w = 1

            image = f.resize(image, size=[h, w], interpolation=f.InterpolationMode.NEAREST)
            mask = f.resize(mask.view(1, mask.shape[-2], mask.shape[-1]), size=[h, w], interpolation=f.InterpolationMode.NEAREST)

        return image, mask


class RandomRotation(FunctionalTransform):

    def __init__(self, rotation_limit: Union[Union[Tuple, List], int],
                 p: float = 0.5,
                 rotation_as_max_pixel_shift: bool = False):
        super(RandomRotation, self).__init__()
        if 0 >= p >= 1:
            raise ValueError(f'Probability must be between 0.0 and 1.0. Got {p}')
        self.p = p
        self.rotation_lim = rotation_limit
        self.max_pixel_shift = rotation_as_max_pixel_shift

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            rotation = self.rotation_lim
            if self.max_pixel_shift:
                if isinstance(self.rotation_lim, (int, float)):

                    if (image.shape[-1] // 2) < self.rotation_lim:
                        rotation = 90
                    else:
                        rotation = np.rad2deg(np.arcsin(self.rotation_lim / (image.shape[-1] / 2)))

                elif isinstance(self.rotation_lim, (Tuple, List)):
                    for i in range(len(self.rotation_lim)):
                        if (image.shape[-1] / 2) > self.rotation_lim[i]:
                            rotation[i] = np.rad2deg(np.arcsin(self.rotation_lim[i] / (image.shape[-1] / 2)))
                        else:
                            rotation[i] = 90

            if isinstance(self.rotation_lim, (int, float)):
                angle = random.uniform(- rotation, rotation)

            elif isinstance(self.rotation_lim, (Tuple, List)):
                angle = random.choice(rotation)

            else:
                raise TypeError(f'Expected angle as +- int or a Sequence of numbers that can be chosen from. '
                                f'Got {type(self.rotation_lim)}')

            image = f.rotate(image, angle, expand=True)
            mask = f.rotate(mask.unsqueeze(0), angle, expand=True).squeeze()

        return image, mask


class RandomPositionShift(FunctionalTransform):

    def __init__(self, p=0.5, max_shift_h: int = 3, max_shift_w: int = 5):
        super(RandomPositionShift, self).__init__()
        self.p = p
        self.max_shift_h = max_shift_h
        self.max_shift_w = max_shift_w

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:

        h, w = image.shape[-2:]

        if torch.rand(1) < self.p:

            h_shift = random.randint(1, self.max_shift_h)
            w_shift = random.randint(1, self.max_shift_w)

            shifted_img = torch.zeros((image.shape[-3], image.shape[-2] + h_shift, image.shape[-1] + w_shift))
            shifted_mask = torch.zeros((image.shape[-3], image.shape[-2] + h_shift, image.shape[-1] + w_shift))

            if torch.rand(1) > 0.5:
                h_shift = - h_shift
            if torch.rand(1) > 0.5:
                w_shift = - w_shift

            if h_shift > 0:
                h_start = h_shift
                h_end = shifted_img.shape[-2]
            else:
                h_start = 0
                h_end = shifted_img.shape[-2] + h_shift

            if w_shift > 0:
                w_start = w_shift
                w_end = shifted_img.shape[-1]
            else:
                w_start = 0
                w_end = shifted_img.shape[-1] + w_shift

            shifted_img[:, h_start:h_end, w_start:w_end] = image
            shifted_mask[:, h_start:h_end, w_start:w_end] = mask

            return shifted_img, shifted_mask

        else:
            return image, mask


class RandomBrightnessAdjust(FunctionalTransform):

    def __init__(self, p: float = 0.5, brightness_range: Tuple[float, float] = (1, 1)):
        super(RandomBrightnessAdjust, self).__init__()
        if 0 >= p >= 1:
            raise ValueError(f'Probability must be between 0.0 and 1.0. Got {p}')
        if any(brightness_range) < 0:
            raise ValueError(f'brightness_range values must be non-negative. Got {brightness_range}')
        if brightness_range[0] > brightness_range[1]:
            raise ValueError(
                f'Brightness value at index 0 must be smaller than that of index 1. Got {brightness_range}')
        self.p = p
        self.brightness = brightness_range

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:

        if torch.rand(1) < self.p:
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            image = f.adjust_brightness(image, brightness_factor)

        return image, mask


class RandomContrastAdjust(FunctionalTransform):

    def __init__(self, p: float = 0.5, contrast_range: Tuple[float, float] = (1, 1)):
        super(RandomContrastAdjust, self).__init__()
        if 0 >= p >= 1:
            raise ValueError(f'Probability must be between 0.0 and 1.0. Got {p}')
        if any(contrast_range) < 0:
            raise ValueError(f'brightness_range values must be non-negative. Got {contrast_range}')
        if contrast_range[0] > contrast_range[1]:
            raise ValueError(f'Contrast value at index 0 must be smaller than that of index 1. Got {contrast_range}')
        self.p = p
        self.brightness = contrast_range

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            contrast_factor = random.uniform(self.brightness[0], self.brightness[1])
            image = f.adjust_contrast(image, contrast_factor)

        return image, mask


class RandomEqualize(FunctionalTransform):

    def __init__(self, p=0.5):
        super(RandomEqualize, self).__init__()
        self.p = p
        self.equalize = f.equalize

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            image = self.equalize(image.contiguous())

        return image, mask



if __name__ == '__main__':
    from imageio.v2 import imread
    from models.unets import UNet
    from torchmetrics.functional.classification import binary_jaccard_index

    torch.manual_seed(42)
    im_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Train\\data\\MAD6400_2008-07-02_arcd_60@vhf_400749_38_95.png'
    msk_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Train\\label\\MAD6400_2008-07-02_arcd_60@vhf_400749_38_95.png'

    resize_const = QuasiResize([64, 64], 2, padding_mode='constant')
    resize_ref = QuasiResize([64, 64], 2, padding_mode='reflect')
    resize_rep = QuasiResize([64, 64], 2, padding_mode='replicate')

    un_resize = UndoQuasiResize(resize_const)
    norm = Normalize((0, 1), (0, 255), return_type=torch.float32)
    float32 = ConvertDtype(torch.float32)

    im = torch.from_numpy(np.array(imread(im_path)).transpose((2, 0, 1)))[:, :, :22].to(torch.float32)
    msk = torch.from_numpy(np.array(imread(msk_path)).transpose((2, 0, 1)))[:, :, :22].to(torch.float32)
    print(im.shape)
    transform = RandomContrastAdjust(1, (1.1, 1.1))

    fig, ax = plt.subplots(4, 2)

    ax[0, 0].imshow(im[0, :, :], cmap='jet', vmin=0, vmax=255)
    ax[0, 1].imshow(msk[0, :, :], cmap='jet', vmin=0, vmax=1)

    ax[1, 0].imshow(resize_const(im)[0, :, :], cmap='jet', vmin=0, vmax=255)
    ax[1, 1].imshow(resize_const(msk)[0, :, :], cmap='jet', vmin=0, vmax=1)

    ax[2, 0].imshow(resize_ref(im)[0, :, :], cmap='jet', vmin=0, vmax=255)
    ax[2, 1].imshow(resize_ref(msk)[0, :, :], cmap='jet', vmin=0, vmax=1)

    ax[3, 0].imshow(resize_rep(im)[0, :, :], cmap='jet', vmin=0, vmax=255)
    ax[3, 1].imshow(resize_rep(msk)[0, :, :], cmap='jet', vmin=0, vmax=1)

    plt.show()


    """
    im1 = norm(im)
    msk = float32(msk)

    im1 = torch.stack((resize(im1[:, :, 0:58]), resize(im1[:, :, 58:58*2]))).cuda()
    msk1 = torch.stack((resize(msk[:, :, 0:58]), resize(msk[:, :, 58:58*2]))).cuda()

    im, msk = transform(im, msk)

    im = norm(im)
    msk = float32(msk)

    im2 = torch.stack((resize(im[:, :, 0:58]), resize(im[:, :, 58:116]))).cuda()
    msk2 = torch.stack((resize(msk[:, :, 0:58]), resize(msk[:, :, 58:116]))).cuda()
    
    model = UNet().cuda()
    pre_trained = torch.load('C:\\Users\\dombe\\PycharmProjects\\Test\\weights\\Unet_32_pretrain-True_loss-BinaryDiceLoss_optim-adam\\lr_0.001_wd_0.1_betas_0.8-0.999_momentum_0.9_freezed-0_0.pt')
    model.init_weights(pre_trained['model_state'])

    model.eval()

    with torch.no_grad():

        pred = model(im1)

        pred = torch.where(pred.squeeze() >= 0.5, 1, 0)
        #pred1 = torch.where(pred1 >= 0.5, 1, 0)

        iou = binary_jaccard_index(pred, msk1[:, 0, :, :]).item()

        fig, ax = plt.subplots(3, 2)
        ax[0, 0].imshow(im1.cpu()[0, 0, :, :], cmap='jet')
        ax[0, 1].imshow(im1.cpu()[1, 0, :, :], cmap='jet')
        ax[1, 0].imshow(msk1.cpu()[0, 0, :, :], cmap='jet', vmin=0, vmax=1)
        ax[1, 1].imshow(msk1.cpu()[1, 0, :, :], cmap='jet', vmin=0, vmax=1)
        ax[2, 0].imshow(pred.cpu()[0, :, :], cmap='jet', vmin=0, vmax=1)
        ax[2, 1].imshow(pred.cpu()[1, :, :], cmap='jet', vmin=0, vmax=1)
        fig.suptitle(f'IoU: {iou}', fontsize=16)
        plt.show()

        pred = model(im2)

        pred = torch.where(pred.squeeze() >= 0.5, 1, 0)
        #pred1 = torch.where(pred1 >= 0.5, 1, 0)

        iou = binary_jaccard_index(pred, msk2[:, 0, :, :]).item()

        fig, ax = plt.subplots(3, 2)
        ax[0, 0].imshow(im2.cpu()[0, 0, :, :], cmap='jet')
        ax[0, 1].imshow(im2.cpu()[1, 0, :, :], cmap='jet')
        ax[1, 0].imshow(msk2.cpu()[0, 0, :, :], cmap='jet', vmin=0, vmax=1)
        ax[1, 1].imshow(msk2.cpu()[1, 0, :, :], cmap='jet', vmin=0, vmax=1)
        ax[2, 0].imshow(pred.cpu()[0, :, :], cmap='jet', vmin=0, vmax=1)
        ax[2, 1].imshow(pred.cpu()[1, :, :], cmap='jet', vmin=0, vmax=1)
        fig.suptitle(f'IoU: {iou}', fontsize=16)
        plt.show()



    """
    """
    #bright_adjust = RandomBrightnessAdjust(0.99, (1.2, 1.2))
    contrast_adjust = RandomContrastAdjust(0.99, (0.7, 0.7))
    flip = RandomHorizontalFlip(0.99)

    norm = Normalize((0, 1), (0, 255))
    fig, ax = plt.subplots(2, 1)

    min_, max_ = torch.min(im), torch.max(im)

    ax[0].imshow(im[0, :, :], cmap='jet', vmin=min_, vmax=max_)
    #im, msk = bright_adjust(im, msk)
    im, msk = contrast_adjust(im, msk)
    print(torch.max(im))
    im = norm(im)
    #im, msk = flip(im, msk)
    ax[1].imshow(im[0, :, :], cmap='jet', vmin=0, vmax=1)
    plt.show()
    """
    """
    transforms = [RandomVerticalFlip(0.9), RandomHorizontalFlip(0.9)]

    objaug = ObjectAugmentation(transforms)

    objaug(im, msk)

    cast = Normalize((0, 1), (0, 255), return_type=torch.float32)
    print(im.dtype)
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(im[0, :, :], cmap='jet')
    im = cast(im)
    print(im.dtype)
    ax[1].imshow(im[0, :, :], cmap='jet')
    print(im)
    plt.show()
    """
