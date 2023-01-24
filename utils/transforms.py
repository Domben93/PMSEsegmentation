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
    "Crop"
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


TransformTypes = TypeVar("TransformTypes", Transform, FunctionalTransform)


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
                    f'For each transform pair it is only allowed with on or two transforms'
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

    def __call__(self, image):
        return f.convert_image_dtype(image, self.dtype)


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
            raise TypeError(f'Image must be type: Tensor. Got {type(image)} instead')
        if not len(image.shape) >= 3:
            raise ValueError(f'Tensor is expected to have [..., H, W] shape. Got {image.shape} instead')
        h, w = image.shape[-2], image.shape[-1]

        # print(h, w)
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
            padding[0] = math.ceil(h_pad_val / 2)  # left padding value
            padding[2] = math.floor(h_pad_val / 2)  # right padding value

        if w_pad_val:
            padding[1] = math.ceil(w_pad_val / 2)  # top padding value
            padding[3] = math.floor(w_pad_val / 2)  # bottom padding value

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
                if o_h * (i + 1) > h:
                    actual_h_scale = i

                if o_w * (i + 1) > w:
                    actual_w_scale = i

                if actual_h_scale != 0 and actual_w_scale != 0:
                    break

            image_w_padding = (w - actual_w_scale * o_w) / 2
            image_h_padding = (h - actual_h_scale * o_h) / 2

            image = image[::, math.ceil(image_h_padding):-math.floor(image_h_padding),
                    math.ceil(image_w_padding):-math.floor(image_w_padding)]

        image = f.resize(image, [o_h, o_w], self.interpolation)

        return image


class Standardize(Transform):

    def __init__(self, mean: Tensor, std: Tensor, inplace: bool = False):
        super(Standardize, self).__init__()

        if not (isinstance(mean, Tensor) or isinstance(std, Tensor)):
            raise TypeError(f'Mean and Tensor must be a Tensor. Got {type(mean)} and {type(std)}')
        if mean.shape != std.shape:
            raise ValueError(f'Mean and Std must have same shape. Got {mean.shape} and {std.shape}')

        self.mean = mean.tolist()
        self.std = std.tolist()

        self.inplace = inplace

    def __call__(self, image: Tensor) -> Tensor:

        if isinstance(image, ndarray):
            image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float64)
        if image.shape[-3] != len(self.mean):
            raise ValueError(
                f'Number of channels in image does not match. Got image shape: {image.shape} and mean and std '
                f'shape: {len(self.mean)} & {len(self.std)}')
        return f.normalize(image, mean=self.mean, std=self.std, inplace=self.inplace)


class Normalize(Transform):

    def __init__(self, min_max_range: Tuple[float, float], dataset_min_max: Tuple[float, float]):
        super(Normalize, self).__init__()
        self.min_range, self.max_range = min_max_range
        self.min = dataset_min_max[0]
        self.max = dataset_min_max[1]

    def __call__(self, image: Union[Tensor]) -> Tensor:
        """
        image = image.numpy().squeeze()
        scaler = preprocessing.MinMaxScaler(feature_range=self.feature_range)
        scaler.fit(image)
        return scaler.transform(image)
        """

        image = image.squeeze()

        X_std = (image - self.min) / (self.max - self.min)
        image = X_std * (self.max_range - self.min_range) + self.min_range

        return image.numpy().astype(np.uint8)


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

    def __call__(self, mask: ndarray) -> ndarray:

        for class_, new_val in zip(self.remove_list, self.replace_val):
            if self.replace_val:
                mask = np.where(mask == class_, new_val, mask)

        return mask.transpose((2, 0, 1))


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

    def __init__(self, p, scale: Sequence, small_size_scaler: bool = True):
        super(RandomResize, self).__init__()
        if 0 >= p >= 1:
            raise ValueError(f'Probability must be between 0.0 and 1.0. Got {p}')
        self.p = p
        self.scale = scale
        self.small_scaler = small_size_scaler

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:

        if torch.rand(1) < self.p:
            if len(self.scale) == 2:
                scale_h = random.uniform(self.scale[0], self.scale[1])
                scale_w = random.uniform(self.scale[0], self.scale[1])

            elif isinstance(self.scale, Sequence):

                scale_h = random.choice(self.scale)
                scale_w = random.choice(self.scale)

            else:
                raise TypeError(f'Expected Sequence of numbers that can be chosen from. '
                                f'Got {type(self.scale)}')
            """
            h, w = image.shape[0:2]
            TODO: implement such that small pixel objects are scaled by a different value
            if self.small_scaler:
                if h < self.small_scaler:
                    scale_h = (scale_h * ratio) / h
    
    
                if w < self.small_scaler:
                    scale_w = (scale_w * ratio) / w
            """

            h, w = int(image.shape[0] * scale_h), int(image.shape[1] * scale_w)

            image = f.resize(image, size=[h, w], interpolation=f.InterpolationMode.NEAREST)
            mask = f.resize(mask, size=[h, w], interpolation=f.InterpolationMode.NEAREST)

        return image, mask


class RandomRotation(FunctionalTransform):

    def __init__(self, rotation_limit: Union[Sequence, int], p: float = 0.5):
        super(RandomRotation, self).__init__()
        if 0 >= p >= 1:
            raise ValueError(f'Probability must be between 0.0 and 1.0. Got {p}')
        self.p = p
        self.rotation_lim = rotation_limit

    def __call__(self, image: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        if torch.rand(1) < self.p:
            if isinstance(self.rotation_lim, int):
                angle = random.randrange(- self.rotation_lim, self.rotation_lim)

            elif isinstance(self.rotation_lim, Sequence):
                angle = random.choice(self.rotation_lim)

            else:
                raise TypeError(f'Expected angle as +- int or a Sequence of numbers that can be chosen from. '
                                f'Got {type(self.rotation_lim)}')

            image = f.rotate(image.view(1, image.shape[-2], [image.shape[-1]]), angle)
            mask = f.rotate(image.view(1, image.shape[-2], [image.shape[-1]]), angle)

        return image, mask


class ObjectAugmentation(Transform):

    def __init__(self, augmentation_methods: List[FunctionalTransform],
                 ):
        super(ObjectAugmentation, self).__init__()

        self.tranforms = augmentation_methods

    @staticmethod
    def _image_parsing(image: Tensor, mask: Tensor) -> Tensor:

        parsed = torch.where(mask == 1, 0, image)
        return parsed

    def _background_inpainting(self, parsed_image: Tensor) -> Tensor:

        defect_image = np.array(parsed_image[:, :, 0])
        mask = torch.where(parsed_image == 0, 1, 0)[:, :, 0]
        mask = self.__mask_padding(mask, 3)

        inpainted = inpaint.inpaint_biharmonic(defect_image, mask.numpy(), channel_axis=None)

        """
        only_inpainted = np.where(mask == 1, inpainted, np.nan) 
        min_, max_ = np.nanmin(only_inpainted), np.nanmax(only_inpainted)
        inpainted_indx = np.where(mask == 1)
        for x, y in zip(inpainted_indx[0], inpainted_indx[1]):

            inpainted[x, y] = random.uniform(float(min_), float(max_))
        """

        return torch.from_numpy(inpainted)

    def _object_augmentation(self, obj_image, obj_mask) -> Tuple[Tensor, Tensor]:

        for transform in self.tranforms:
            obj_image, obj_mask = transform(obj_image, obj_mask)

        return obj_image, obj_mask

    def __call__(self, image: Tensor, mask: Tensor):

        new_mask = torch.zeros(mask.shape)
        image1 = self._image_parsing(image, mask)
        inpainted_image = self._background_inpainting(image1) * 255
        inpainted_image_orig = inpainted_image
        img_mask, island_count = measure.label(mask, background=0, return_num=True, connectivity=1)

        #img_islands, mask_islands, bounding_box = [], [], []
        for i in range(island_count - 1):

            pixel_island = torch.where(torch.from_numpy(img_mask) == i + 1)

            x, y = torch.min(pixel_island[0]), torch.min(pixel_island[1])
            x2, y2 = torch.max(pixel_island[0]), torch.max(pixel_island[1])

            im, msk = self._object_augmentation(image[x:x2, y:y2], mask[x:x2, y:y2])

            inpainted_image[x:x2, y:y2] = torch.where(msk[:, :, 0] == 1, im[:, :, 0], inpainted_image[x:x2, y:y2])
            new_mask[x:x2, y:y2] = msk


        fig, ax = plt.subplots(1, 6)
        min, max = torch.min(image[:, :, 0]), torch.max(image[:, :, 0])

        ax[0].imshow(image[:, :, 0], cmap='jet', vmin=min, vmax=max, label='Original Image')
        ax[1].imshow(image1[:, :, 0], cmap='jet', vmin=min, vmax=max, label='Parsed image')
        ax[2].imshow(inpainted_image_orig, cmap='jet', vmin=min, vmax=max, label='Inpainted')
        ax[3].imshow(mask[:, :, 0], cmap='jet', label='Original mask')

        ax[4].imshow(inpainted_image, cmap='jet', vmin=min, vmax=max, label='Assemble Image')
        ax[5].imshow(new_mask[:, :, 0], cmap='jet', vmin=min, vmax=max, label='Assemble Mask')
        plt.show()


    def __mask_padding(self, mask, value: int = 1) -> Tensor:
        mask_copy = torch.zeros(mask.shape)

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x, y] == 1:
                    mask_copy[max(x - value, 0):min(mask.shape[0], value + x),
                              max(y - value, 0):min(mask.shape[1], value + y)] = 1
        return mask_copy


if __name__ == '__main__':
    from imageio.v2 import imread

    im_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Train\\data\\MAD6400_2011-06-01_manda_59_0_53.png'
    msk_path = 'C:\\Users\\dombe\\PycharmProjects\\Test\\dataset\\Train\\label\\MAD6400_2011-06-01_manda_59_0_53.png'
    im = torch.from_numpy(np.array(imread(im_path)))
    msk = torch.from_numpy(np.array(imread(msk_path)))

    transforms = [RandomVerticalFlip(0.2), RandomHorizontalFlip(0.2)]

    objaug = ObjectAugmentation(transforms)

    objaug(im, msk)
