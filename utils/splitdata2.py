import math
import os
import random
import warnings
from utils import *
from typing import Tuple, Union, Dict, Any, NoReturn
import numpy as np
from tqdm import tqdm
from numpy import ndarray
from pathlib import Path
from collections import OrderedDict
import cv2


class SplitSegData:
    """
    Data splitter that splits images into specified sizes or shapes
    """

    def __init__(self, image_dir: str,
                 label_dir: str,
                 horizontal_split: Tuple[int, bool] = None,
                 vertical_split: Tuple[int, bool] = None,
                 square_split: bool = False,
                 split_overlap: float = .0,
                 last_min_overlap: Union[int, float] = 0,
                 get_trailing: bool = False,
                 **kwargs):
        """
        Split segmentation data.
        Args:
            image_dir: image dir path
            label_dir: label dir path
            horizontal_split: Specify horizontal split size
            vertical_split: Specify vertical split size
            square_split: Square split images. Results in images where width is equal to height. Overrides
            horizontal_split and vertical_split.
            split_overlap: Percentage overlap between samples [0-1]
            last_min_overlap: minimum overlap of last before including in split Union[int, float].
            Either int (num pixels) or float (percentage 0-1)
            get_trailing: Overrides last_min_overlap (default: False)
            **kwargs:
             - disable_warning: bool: Suppresses warnings that might be caused if True
        """

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.horizontal_split = horizontal_split[0] if horizontal_split is not None else None
        self.vertical_split = vertical_split[0] if vertical_split is not None else None
        self.horizontal_overlap = horizontal_split[1] if self.horizontal_split is not None else False
        self.vertical_overlap = vertical_split[1] if self.vertical_split is not None else False

        self.percentage_overlap = split_overlap
        self.square_split = square_split
        self.split_save: bool = False
        self.split_save_dir: str = ''

        self.get_trailing = get_trailing

        if isinstance(last_min_overlap, float):
            assert 0 < last_min_overlap <= 1.0, 'If minimum overlap if given as float it must be in the interval(0,1]' \
                                                '(percentage of the split size that is the minimum allowed last overlap).'

            self.last_min_overlap = last_min_overlap
            self.horizontal_overlap = True

        elif isinstance(last_min_overlap, int) and last_min_overlap >= 1:
            self.last_min_overlap = last_min_overlap
            self.horizontal_overlap = True

        elif self.get_trailing:
            self.horizontal_overlap = True
            self.last_min_overlap = 0
        # else:
        #    raise TypeError(f'last_min_overlap must be of type Int of float (on interval (0, 1]).'
        #                    f'Got {type(last_min_overlap)}.')

        if kwargs.get('disable_warnings') is True:
            warnings.filterwarnings('ignore')

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f'Could not find: {self.image_dir}')
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f'Could not find: {self.label_dir}')

        if self.horizontal_split is None and self.vertical_split is None and square_split is False:
            warnings.warn('Redundant object instantiation as no split is chosen!\n'
                          'Data will have the same structure as in the specified Image and label folder.')

        self.split_info = self._split_data_square()

    def __len__(self):
        number_of_images = 0
        for img in list(self.split_info.keys()):
            number_of_images += self.split_info[img]['number_of_splits']
        return number_of_images

    def __getitem__(self, item: int) -> Tuple[Any, Any, Any]:

        if not isinstance(item, int):
            raise TypeError(f'index must be a integer. Got {type(item)}')

        if item >= len(self):
            raise ValueError(f'index value must be >= number of images {len(self)}. Got {item}')

        if self.split_save:

            img_path = os.path.join(self.split_save_dir, 'data')
            mask_path = os.path.join(self.split_save_dir, 'label')

            if not os.path.isdir(self.split_save_dir):
                raise NotADirectoryError(f'{self.split_save_dir} must be a directory or folder')
            if not os.path.isdir(img_path):
                raise NotADirectoryError(f'"{img_path}" must be a directory or folder')
            if not os.path.isdir(mask_path):
                raise NotADirectoryError(f'"{mask_path}" must be a directory or folder')

            images_ = os.listdir(img_path)
            masks_ = os.listdir(mask_path)

            image = default_loader(os.path.join(img_path, images_[item]), return_type=ndarray)
            mask = default_loader(os.path.join(mask_path, masks_[item]), return_type=ndarray)
            name = Path(images_[item]).stem

            return image, mask, name

        img_path = os.path.join(self.image_dir)
        mask_path = os.path.join(self.label_dir)

        images_ = os.listdir(img_path)
        masks_ = os.listdir(mask_path)

        global_idx = item

        for i, img_name in enumerate(list(self.split_info.keys())):

            local_size = self.split_info[img_name]['number_of_splits']

            if global_idx >= local_size:
                global_idx -= local_size
                continue

            else:
                split = self.split_info[img_name]['splits'][global_idx]

                (lr, rr), (lc, rc) = split

                image = default_loader(os.path.join(self.image_dir, images_[i]))[lr:rr, lc:rc]
                mask = default_loader(os.path.join(self.label_dir, masks_[i]))[lr:rr, lc:rc]

                info = {'image_name': img_name,
                        'split_info': split}

                return image, mask, info

    def get_split(self):
        return self.split_info

    def _split_data_square(self) -> Dict:

        image_list = sorted(os.listdir(self.image_dir))
        mask_list = sorted(os.listdir(self.label_dir))
        image_name_keys = []

        horizontal_pixel_overlap = int(self.horizontal_split - self.horizontal_split * (1 - self.percentage_overlap)) \
            if self.horizontal_split is not None else 0

        for im in image_list:
            image_name_keys.append(Path(im).stem)
        image_name_keys.sort()

        split_info = OrderedDict()

        if len(image_list) != len(mask_list):
            raise ValueError(f'Number of Images ({len(image_list)}) != number of mask ({len(mask_list)})')

        for im_name, im, msk in tqdm(zip(image_name_keys, image_list, mask_list), desc='Splitting images and mask pair',
                                     total=len(image_list)):

            image = default_loader(os.path.join(self.image_dir, im), return_type=ndarray)
            mask = default_loader(os.path.join(self.label_dir, msk), return_type=ndarray)

            assert isinstance(image, ndarray)
            assert isinstance(mask, ndarray)

            if image.shape != mask.shape:
                raise ValueError(f'Mismatch between image shape ({image.shape} and mask shape ({mask.shape})')

            if self.square_split:
                self.horizontal_split = self.vertical_split = min(image.shape[0], image.shape[1])
                horizontal_pixel_overlap = int(
                    self.horizontal_split - self.horizontal_split * (1 - self.percentage_overlap))

            num_horizontal_split = self.__verify_split(image.shape[1], self.horizontal_split, horizontal_pixel_overlap)
            num_vertical_splits = self.__verify_split(image.shape[0], self.vertical_split)

            """
            if self.percentage_overlap:
                num_col_splits *= (1 + self.percentage_overlap)
                print(num_col_splits)
                num_col_splits = int(num_col_splits)

                horizontal_pixel_overlap = int(self.column_split - self.column_split * (1 - self.percentage_overlap))
            """
            """
            if float(num_col_splits or 0) and float(num_vertical_splits or 0) <= 1:
                warnings.warn(f'Column and Row split value is equal or larger than image and mask shape.\n'
                              f'No action is made and original image and mask shape ({image.shape}) will be set as shape'
                              f'in returned dict')
            """

            image_split = OrderedDict()
            right_row, left_row, left_col, right_col = 0, 0, 0, 0
            row, col, horizontal_overlap, vertical_overlap = 0, 0, 0, 0
            num = 0

            for row in range(num_vertical_splits):
                left_row = right_row
                if image.shape[0] <= int(self.vertical_split or image.shape[0]):
                    # warnings.warn(f'Row split chosen is bigger or equal to image rows.\n'
                    #              f'Image and mask ({im_name}) will keep original row shape')
                    right_row = image.shape[0]
                else:
                    right_row = left_row + self.vertical_split

                right_col, left_col = 0, 0

                for col in range(num_horizontal_split):
                    left_col = right_col - horizontal_pixel_overlap if right_col > 0 else 0
                    if image.shape[1] <= int(self.horizontal_split or image.shape[1]):

                        right_col = image.shape[1]
                        # warnings.warn(f'Column split chosen is bigger or equal to image columns.\n'
                        #              f'Image and mask ({im_name}) will keep original column shape')
                    else:
                        right_col = left_col + self.horizontal_split

                    image_split.update({num: [(left_row, right_row), (left_col, right_col)]})
                    num += 1

                if self.horizontal_overlap:
                    # horizontal_overlap = image.shape[1] % ((self.horizontal_split - horizontal_pixel_overlap) *
                    #                            (num_horizontal_split - 1) + self.horizontal_split)

                    horizontal_overlap = self.__last_overlap(image.shape[1],
                                                             self.horizontal_split,
                                                             horizontal_pixel_overlap,
                                                             num_horizontal_split)

                    if horizontal_overlap:

                        if self.get_trailing:

                            left_col = right_col
                            right_col = left_col + horizontal_overlap

                        else:
                            left_col = left_col - horizontal_pixel_overlap + horizontal_overlap
                            right_col = right_col - horizontal_pixel_overlap + horizontal_overlap

                        image_split.update({num: [(left_row, right_row), (left_col, right_col)]})
                        num += 1

            if self.vertical_overlap:

                vertical_overlap = image.shape[0] % self.vertical_split
                if vertical_overlap:
                    left_row = left_row + vertical_overlap
                    right_row = left_row + self.vertical_split + vertical_overlap

                    image_split.update({num: [(left_row, right_row), (left_col, right_col)]})
                    num += 1

            meta_data = self.__generate_metadata(image, num, image_split, row, col, horizontal_overlap,
                                                 vertical_overlap)

            split_info.update({im_name: meta_data})

        return split_info

    def save_split_to_folder(self, path: str, extension: str = None):

        if not os.path.exists(path):
            raise FileNotFoundError(f'Could not find: {path}')

        if os.path.exists(path):
            raise FileExistsError(f'No such folder exists: "{path}"')

        os.mkdir(path)

        for image in list(self.split_info.keys()):
            split_list = list(image.keys())
            for split in split_list:
                print(split)

    @staticmethod
    def __generate_metadata(im, num, im_split, row, col, h_overlap, v_overlap) -> Dict:
        meta_data = {
            'original_size': im.shape,
            'number_of_splits': num,
            'split_array': np.reshape(np.arange(1, num + 1), (row + 1 + (1 if v_overlap else 0),
                                                              col + 1 + (1 if h_overlap else 0))),
            'splits': im_split
        }
        return meta_data

    @staticmethod
    def __verify_split(split_shape, split_val, overlap=0) -> Union[int, float]:
        try:
            if overlap:
                split_shape = split_shape - split_val
                split_val = split_val - overlap
                return 1 + split_shape // split_val if (1 < split_val < split_shape) else 1
            else:
                return split_shape // split_val if (1 < split_val < split_shape) else 1
        except:
            return 1

    def __last_overlap(self, im_direction_shape, split, pixel_overlap, number_of_splits):

        overlap = im_direction_shape % ((split - pixel_overlap) * (number_of_splits - 1) + split)

        if isinstance(self.last_min_overlap, int) and overlap >= self.last_min_overlap:
            return overlap

        elif isinstance(self.last_min_overlap, float):
            if overlap >= int(self.last_min_overlap * split):
                return overlap
            else:
                return 0
        else:
            return 0


class StratifiedSegDataSplit:

    def __init__(self,
                 image_dir: str,
                 label_dir: str,
                 train_val_test_split: float = 0.6,
                 random_split: bool = True,
                 random_seed: int = 42):
        """
        Stratified split of segmentation data.
        Splits each image or array by column into train, val, and test data.
        Args:
            image_dir: Path to all images/data arrays
            label_dir: Path to all label masks
            train_val_test_split: Split between 0-1. Validation and test size will automatically be set equal.
            (e.g. if train_val_test_split is set to 0.6 then val and test sice will be 0.2 for both)
            random_split: Toggle random split. If false, Train will e.g. have the first 0.6*total pixels, validation
            will have the next 0.2*total pixels, and then test the last 0.2. If set TRUE, the first, second and third
            selection is random.
            random_seed: Random seed for reproducibility
        """

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.split = train_val_test_split
        self.train_split = self.split
        self.random_split = random_split
        self.random_seed = random_seed
        if self.random_split:
            random.seed(self.random_seed)

        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f'Could not find: {self.image_dir}')
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f'Could not find: {self.label_dir}')

        if self.split > 1:
            raise ValueError(f'Train split must be below 1.0. Got {self.train_split}.')

        self.split_info = self._split()

    def _split(self):

        image_list = sorted(os.listdir(self.image_dir))
        mask_list = sorted(os.listdir(self.label_dir))
        image_name_keys = []

        for im in image_list:
            image_name_keys.append(Path(im).stem)
        image_name_keys.sort()

        split_info = OrderedDict()

        keys_ = ['train', 'validation', 'test']
        shuffled_keys = keys_
        if len(image_list) != len(mask_list):
            raise ValueError(f'Number of Images ({len(image_list)}) != number of mask ({len(mask_list)})')

        for im_name, im, msk in tqdm(zip(image_name_keys, image_list, mask_list), desc='Splitting images and mask pair',
                                     total=len(image_list)):

            image = default_loader(os.path.join(self.image_dir, im), return_type=ndarray)
            mask = default_loader(os.path.join(self.label_dir, msk), return_type=ndarray)

            assert isinstance(image, ndarray)
            assert isinstance(mask, ndarray)

            if image.shape != mask.shape:
                raise ValueError(f'Mismatch between image shape ({image.shape} and mask shape ({mask.shape})')

            split_info[im_name] = {}
            dict_ = {}.fromkeys(keys_, None)

            split_vals = self.__get_split_sizes(image)

            for split_val, key in zip(split_vals, dict_.keys()):
                dict_[key] = split_val

            if self.random_split:
                shuffled_keys = random.sample(keys_, k=len(keys_))

            shift = 0
            for d_key in shuffled_keys:

                split_info[im_name][d_key] = (shift, shift + dict_[d_key])
                shift += dict_[d_key]
        print(split_info)
        return split_info

    def save(self, save_dir,
             specific: str = 'train',
             img_transforms=None,
             mask_transforms=None) -> NoReturn:

        if not os.path.exists(save_dir):
            raise FileNotFoundError

        folders = ['Train', 'Validation', 'Test']
        for folder in folders:
            if not os.path.exists(os.path.join(save_dir, folder)):
                raise FileNotFoundError
            elif not os.path.join(save_dir, folder, 'data'):
                raise FileNotFoundError
            elif not os.path.join(save_dir, folder, 'label'):
                raise FileNotFoundError
            else:
                if os.listdir(os.path.join(save_dir, folder, 'data')) != 0:
                    for f in os.listdir(os.path.join(save_dir, folder, 'data')):
                        os.remove(os.path.join(os.path.join(save_dir, folder, 'data'), f))
                if os.listdir(os.path.join(save_dir, folder, 'label')) != 0:
                    for f in os.listdir(os.path.join(save_dir, folder, 'label')):
                        os.remove(os.path.join(os.path.join(save_dir, folder, 'label'), f))

        if specific not in ['train', 'validation', 'test', 'all']:
            raise ValueError(f'specific variable must be set to either "train", "validation" or "test", Got {specific}.')

        if specific == 'all':
            specific = ['train', 'validation', 'test']

        for i in range(len(self)):

            if isinstance(specific, list):
                for specif in specific:
                    im, msk, name = self[i, specif]
                    im = np.array([im])
                    msk = np.array([msk])

                    im = im.transpose((1, 2, 0))
                    msk = msk.transpose((1, 2, 0))

                    if img_transforms:
                        im = img_transforms(im)
                    if mask_transforms:
                        msk = mask_transforms(msk)

                    im = im.numpy().transpose((1, 2, 0))
                    msk = msk.numpy().transpose((1, 2, 0))

                    cv2.imwrite(save_dir + '/' + specif + '/data' + '/' + name + '.png', im)
                    cv2.imwrite(save_dir + '/' + specif + '/label' + '/' + name + '.png', msk)
            else:
                im, msk, name = self[i, specific]
                im = np.array([im])
                msk = np.array([msk])

                im = im.transpose((1, 2, 0))
                msk = msk.transpose((1, 2, 0))

                if img_transforms:
                    im = img_transforms(im)
                if mask_transforms:
                    msk = mask_transforms(msk)

                cv2.imwrite(save_dir + '/' + specific + '/data' + '/' + name + '.png', im)
                cv2.imwrite(save_dir + '/' + specific + '/label' + '/' + name + '.png', msk)

    def __generate_metadata(self, img_name: str, split: Dict):
        raise NotImplementedError

    def __get_split_sizes(self, image: ndarray) -> Tuple[int, int, int]:

        im_columns = image.shape[1]

        num_train = math.ceil(im_columns * self.train_split)
        num_val = math.ceil((im_columns - num_train) / 2)
        num_test = im_columns - num_train - num_val

        return num_train, num_val, num_test

    def __len__(self) -> int:
        return len(list(self.split_info.keys()))

    def __getitem__(self, item: Tuple[int, str]) -> Tuple[ndarray, ndarray, str]:

        specific = item[1]

        if not isinstance(item[0], int):
            raise TypeError(f'index must be a integer. Got {type(item)}')

        if item[0] >= len(self):
            raise ValueError(f'index value must be >= number of images {len(self)}. Got {item}')

        if specific not in ['train', 'validation', 'test']:
            raise ValueError(f'specific variable must be set to either "train", "validation" or "test", Got {specific}.')

        image_name = os.listdir(self.image_dir)[item[0]]
        mask_name = os.listdir(self.label_dir)[item[0]]

        split_l, split_r = self.split_info[Path(image_name).stem][specific]

        img = default_loader(os.path.join(self.image_dir, image_name), return_type=ndarray)[:, split_l:split_r]
        mask = default_loader(os.path.join(self.label_dir, mask_name), return_type=ndarray)[:, split_l:split_r]

        image_name = image_name.split('.')[0] + f'_{split_l}_{split_r}'

        return img, mask, image_name


if __name__ == '__main__':
    import transforms as t
    path_data = '../dataset/Complete/data'
    path_label = '../dataset/Complete/label'

    strat_splitter = StratifiedSegDataSplit(image_dir=path_data, label_dir=path_label)
    strat_splitter.save('../dataset', 'all', )
