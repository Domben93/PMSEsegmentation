import os
import warnings
from .utils import *
from typing import Optional, Tuple, Union, Dict, Any, List, Iterator

import numpy as np

from tqdm import tqdm
from numpy import ndarray
from pathlib import Path


class SplitSegData:
    """
    Data splitter than splits images into specified sizes or shapes
    """
    def __init__(self, image_dir: str,
                 label_dir: str,
                 x_split: Tuple[int, bool] = None,
                 y_split: Tuple[int, bool] = None,
                 square_split: bool = False,
                 split_overlap: float = .0,
                 **kwargs):

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.column_split = x_split[0] if x_split is not None else None
        self.row_split = y_split[0] if y_split is not None else None
        self.column_overlap = x_split[1] if self.column_split is not None else False
        self.row_overlap = y_split[1] if self.row_split is not None else False

        self.percentage_overlap = split_overlap
        self.square_split = square_split
        self.split_save: bool = False
        self.split_save_dir: str = ''

        if kwargs.get('disable_warnings') is True:
            warnings.filterwarnings('ignore')
        if not os.path.exists(self.image_dir):
            raise FileNotFoundError(f'Could not find: {self.image_dir}')
        if not os.path.exists(self.label_dir):
            raise FileNotFoundError(f'Could not find: {self.label_dir}')

        if self.column_split is None and self.row_split is None and square_split is False:
            warnings.warn('Redundant object instantiation as no split is chosen!\n'
                          'Data will have the same structure as in the specified Image and label folder.')

        self.split_info = self._split_data()

    def __len__(self):
        number_of_images = 0
        for img in list(self.split_info.keys()):
            number_of_images += self.split_info[img]['ImagesAfterSplit']
        return number_of_images

    def __getitem__(self, items) -> Tuple[List[Any], List[Any], List[str]]:

        images, masks, names = [], [], []

        if isinstance(items, int):
            items = (items,)
        if not all(item <= len(self) for item in items):
            # vals = [[num for num in items if num >= len(self)]]
            raise ValueError(f'Trying to index items with number(s) larger than the number of images ({len(self)})')

        if self.split_save:

            img_path = os.path.join(self.split_save_dir, 'images')
            mask_path = os.path.join(self.split_save_dir, 'masks')

            if not os.path.isdir(self.split_save_dir):
                raise NotADirectoryError(f'{self.split_save_dir} must be a directory or folder')
            if not os.path.isdir(img_path):
                raise NotADirectoryError(f'{self.split_save_dir} must be a directory or folder')
            if not os.path.isdir(mask_path):
                raise NotADirectoryError(f'{self.split_save_dir} must be a directory or folder')

            images_ = os.listdir(img_path)
            masks_ = os.listdir(mask_path)

            for i in items:
                images.append(default_loader(os.path.join(img_path, images_[i]), return_type=ndarray))
                masks.append(default_loader(os.path.join(mask_path, masks_[i]), return_type=ndarray))
                names.append(Path(images_[i]).stem)
            return images, masks, names

        img_path = os.path.join(self.image_dir)
        mask_path = os.path.join(self.label_dir)

        images_ = os.listdir(img_path)
        masks_ = os.listdir(mask_path)
        split_num = 0
        items = iter(items)
        split_idx = next(items, False)
        current_split_num = 0

        for i, img in enumerate(list(self.split_info.keys())):
            split_num += self.split_info[img]['ImagesAfterSplit']
            while split_num > split_idx:
                for row_split in self.split_info[img]['RowSplit']:
                    for col_split in self.split_info[img]['ColumnSplit']:
                        if current_split_num == split_idx:
                            images.append(default_loader(os.path.join(self.image_dir, images_[i]))
                                          [row_split[0]:row_split[1], col_split[0]:col_split[1]])
                            masks.append(default_loader(os.path.join(self.label_dir, masks_[i]))
                                         [row_split[0]:row_split[1], col_split[0]:col_split[1]])
                            split_name = str(row_split[0]) + 'x' + str(row_split[1]) + '-' + \
                                         str(col_split[0]) + 'x' + str(col_split[1])

                            names.append(img + '_' + split_name)
                            split_idx = next(items, False)
                        current_split_num += 1

                if not split_idx:
                    return images, masks, names

    def get_split(self):
        return self.split_info

    def _split_data(self) -> Dict[str, Dict[str, Optional[Any]]]:

        if not os.path.isdir(self.image_dir):
            raise RuntimeError(f'{self.image_dir} must be a folder or directory')
        if not os.path.isdir(self.label_dir):
            raise RuntimeError(f'{self.label_dir} must be a folder or directory')

        image_list = sorted(os.listdir(self.image_dir))
        mask_list = sorted(os.listdir(self.label_dir))
        image_name_keys = []
        overlap_im_count = 0
        pixel_overlap = int(self.column_split - self.column_split * (1 - self.percentage_overlap))\
            if self.column_split is not None else 0

        for im in image_list:
            image_name_keys.append(Path(im).stem)
        image_name_keys.sort()

        image_info_list = ['RowSplit', 'ColumnSplit', 'ImagesAfterSplit', 'RowOverlap', 'ColOverlap']
        type_list = [list, list, None, None, None]

        split_info = CustomOrderedDict.fromkeys(image_name_keys,
                                                      CustomOrderedDict.fromkeys(image_info_list, type_list))

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
                self.column_split = self.row_split = min(image.shape[0], image.shape[1])
                pixel_overlap = int(self.column_split - self.column_split * (1 - self.percentage_overlap))

            num_col_splits = self.__verify_split(image.shape[1], self.column_split, pixel_overlap)
            num_row_splits = self.__verify_split(image.shape[0], self.row_split)


            """
            if self.percentage_overlap:
                num_col_splits *= (1 + self.percentage_overlap)
                print(num_col_splits)
                num_col_splits = int(num_col_splits)

                pixel_overlap = int(self.column_split - self.column_split * (1 - self.percentage_overlap))
            """
            """
            if float(num_col_splits or 0) and float(num_row_splits or 0) <= 1:
                warnings.warn(f'Column and Row split value is equal or larger than image and mask shape.\n'
                              f'No action is made and original image and mask shape ({image.shape}) will be set as shape'
                              f'in returned dict')
            """

            for row in range(num_row_splits):
                if image.shape[0] <= int(self.row_split or image.shape[0]):
                    split_info[im_name]['RowSplit'].append((0, image.shape[0]))
                    # warnings.warn(f'Row split chosen is bigger or equal to image rows.\n'
                    #              f'Image and mask ({im_name}) will keep original row shape')
                else:
                    split_info[im_name]['RowSplit'].append((row * self.row_split, (row + 1) * self.row_split))

                for col in range(num_col_splits):
                    if image.shape[1] <= int(self.column_split or image.shape[1]):
                        split_info[im_name]['ColumnSplit'].append((0, image.shape[1]))
                        # warnings.warn(f'Column split chosen is bigger or equal to image columns.\n'
                        #              f'Image and mask ({im_name}) will keep original column shape')
                        break
                    else:
                        split_info[im_name]['ColumnSplit'].append(
                            (col * self.column_split - (pixel_overlap * col),
                             (col + 1) * self.column_split - (pixel_overlap * col)))

                if self.column_overlap:
                    overlap = image.shape[1] % ((self.column_split - pixel_overlap) *
                                                (num_col_splits - 1) + self.column_split)
                    print(overlap)
                    if overlap:
                        overlap_im_count += 1
                        split_info[im_name]['ColumnSplit'].append(
                            ((col * self.column_split) - (pixel_overlap * col) + overlap,
                             (col + 1) * self.column_split - (pixel_overlap * col) + overlap))
            if self.row_overlap:
                overlap = image.shape[0] % self.row_split
                if overlap:
                    overlap_im_count += 1
                    split_info[im_name]['RowSplit'].append(((row * self.row_split + overlap),
                                                            ((row + 1) * self.row_split) + overlap))

            split_info[im_name]['ImagesAfterSplit'] = (col + 1) * (row + 1) + overlap_im_count
            overlap_im_count = 0
            split_info[im_name]['RowOverlap'] = self.row_overlap
            split_info[im_name]['ColOverlap'] = self.column_overlap

        return split_info

    def save_split_to_folder(self, path: str, folder_name: str, extension: str = None):
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f'Could not find: {path}')

        if os.path.exists(os.path.join(path, folder_name)):
            raise FileExistsError(f'Folder "{folder_name}" already exists at "{path}"')

        os.mkdir(os.path.join(path, folder_name))

        for image in list(self.split_info.keys()):
            split_list = list(image.keys())
            for split in split_list:
                pass
        """
        raise NotImplementedError('function is not implemented')

    def __verify_split(self, split_shape, split_val, overlap=0) -> Union[int, float]:
        try:
            if overlap:
                split_shape = split_shape - split_val
                split_val = split_val - overlap
                return 1 + split_shape // split_val if (1 < split_val < split_shape) else 1
            else:
                return split_shape // split_val if (1 < split_val < split_shape) else 1
        except:
            return 1

"""
if __name__ == '__main__':
    path_data = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\Array_data_2'
    path_label = 'C:\\Users\\dombe\\Few-shot segmentation\\PMSE-Data\\PMSE_data\\Labels'

    data_splitter = SplitSegData(image_dir=path_data, label_dir=path_label, square_split=True, split_overlap=.3)
    # print(data_splitter.split_info)
    print(len(data_splitter))
    # print(next(islice(data_splitter.split_info.items(), 0, None)))
    # print(data_splitter.split_info[next(islice(data_splitter.split_info, 0, None))])
    #imgs, masks, names = data_splitter[0, 1, 5, 6]
    print(data_splitter.split_info)
"""