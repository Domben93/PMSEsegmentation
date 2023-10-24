import math
import os
import random
from typing import List, Union
import cv2
import numpy as np
import torch
import torch
from torch import Tensor
from torchvision.transforms.functional import rotate
from random import randint
from torchvision import transforms, utils
from tqdm import tqdm
from .transforms import *
from .dataset import PMSE_Dataset


class MaskGenerator:

    def __init__(self,
                 rand_seed=None,
                 figs_max_size: List = None,
                 square_rotation_range: List = None,
                 pad_range: int = 2):

        self.rotation_range = square_rotation_range
        self.max_size = figs_max_size
        self.pad_range = pad_range
        # Seed for reproducibility
        if rand_seed:
            random.seed(rand_seed)

    def generate_inpainting_mask(self, mask: torch.Tensor,
                                 number: int = 200,
                                 max_iter: int = 100,
                                 number_as_ratio: bool = True):

        if self.pad_range:
            mask = self._mask_padding(mask, self.pad_range)

        h, w = mask.shape

        if number_as_ratio:
            num_figs = (h * w) // number
        else:
            num_figs = number

        if isinstance(self.max_size[0], float) and self.max_size[0] < 1:
            max_height = math.ceil(h * self.max_size[0])
        else:
            max_height = self.max_size[0]
        if isinstance(self.max_size[1], float) and self.max_size[1] < 1:
            max_width = math.ceil(w * self.max_size[1])
        else:
            max_width = self.max_size[1]

        rectangles = self._generate_random_rectangle(num_figs, [max_height, max_width], [2, 2])

        if self.rotation_range:
            for idx, rect in enumerate(rectangles):

                rectangles[idx] = torch.squeeze(
                    rotate(rect.view((1, *rect.shape)), angle=random.randint(self.rotation_range[0],
                                                                             self.rotation_range[1]),
                           expand=True))

        all_figure = rectangles

        inpaint_mask = torch.ones(mask.shape)

        for fig_shape in all_figure:

            r_h, r_w = fig_shape.shape

            h_im_index = random.randint(0, h - r_h - 1)
            w_im_index = random.randint(0, w - r_w - 1)

            for i in range(max_iter):

                intersection = torch.sum(mask[h_im_index:(h_im_index + r_h), w_im_index:(w_im_index + r_w)] * fig_shape)

                if intersection == 0:

                    inpaint_mask[h_im_index:(h_im_index + r_h), w_im_index:(w_im_index + r_w)] *= torch.logical_not(fig_shape)
                    break
                else:

                    dir = -1 if random.random() > 0.5 else 1

                    if 0 < h_im_index + dir and h_im_index + r_h + dir > h:
                        x_dir_intersection = \
                            torch.sum(mask[(h_im_index + dir):(h_im_index + r_h + dir),
                                      w_im_index:(w_im_index + r_w)] * fig_shape)
                        if x_dir_intersection < intersection:
                            if 0 < h_im_index + dir > h - r_h:
                                h_im_index += dir
                            elif 0 < h_im_index - dir > h - r_h:
                                h_im_index -= dir

                    if 0 < w_im_index + dir and w_im_index + r_w + dir > w:
                        y_dir_intersection = \
                            torch.sum(mask[h_im_index:(h_im_index + r_h),
                                      (w_im_index + dir):(w_im_index + r_w + dir)] * fig_shape)
                        if y_dir_intersection < intersection:
                            if 0 < w_im_index + dir > w - r_w:
                                w_im_index += dir

                            elif 0 < w_im_index - dir > w - r_w:
                                w_im_index -= dir

        return inpaint_mask

    @staticmethod
    def _generate_random_rectangle(num: int, max: List, min: List) -> List:

        rectangles = []

        for _ in range(num):
            h = random.randint(min[0], max[0])
            w = random.randint(min[1], max[1])
            rectangles.append(torch.ones([h, w]))

        return rectangles

    @staticmethod
    def _generate_random_circles(num: int, max_r: int, min_r: int) -> List:
        circles = []
        for _ in range(num):
            radius = randint(min_r, max_r)
            img = np.zeros(((radius * 2) + 1, (radius * 2) + 1))
            circle = cv2.circle(img, (radius, radius), radius, (1, 1, 1), -1)
            circles.append(torch.from_numpy(circle))
        return circles

    @staticmethod
    def _mask_padding(mask, value: int = 1):
        mask_copy = torch.zeros(mask.shape)

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x, y] == 1:
                    mask_copy[max(x - value, 0):min(mask.shape[0], value + x),
                    max(y - value, 0):min(mask.shape[1], value + y)] = 1
        return mask_copy


def save_generated_masks(train_path, val_path, num=200, masks_pr_image=20, max_sizes: List = None, seed=42):
    """

    :param train_path:
    :param val_path:
    :param masks_pr_image:
    :return:

    Args:
        max_sizes:
    """
    mask_gen = MaskGenerator(rand_seed=seed, figs_max_size=max_sizes, pad_range=2)

    dataset_train = PMSE_Dataset(os.path.join(train_path, 'data'), os.path.join(train_path, 'label'))
    dataset_val = PMSE_Dataset(os.path.join(val_path, 'data'), os.path.join(val_path, 'label'))

    if len(dataset_train) != len(dataset_val):
        raise ValueError('datasets must be of equal length')

    for i in tqdm(range(masks_pr_image), desc='Generating mask for train and validation'):

        for n in range(len(dataset_train)):
            tr_im, tr_msk, tr_info = dataset_train[n]
            vl_im, vl_msk, vl_info = dataset_val[n]

            tr_msk = mask_gen.generate_inpainting_mask(torch.squeeze(tr_msk), number=num)
            vl_msk = mask_gen.generate_inpainting_mask(torch.squeeze(vl_msk), number=num)

            tr_msk = np.array(tr_msk.repeat(3, 1, 1))
            vl_msk = np.array(vl_msk.repeat(3, 1, 1))

            cv2.imwrite(train_path + '/generated_masks' + '/' + tr_info['image_name'] + f'_{i}' + '.png',
                        tr_msk.transpose((1, 2, 0)) * 255)
            cv2.imwrite(val_path + '/generated_masks' + '/' + vl_info['image_name'] + f'_{i}' + '.png',
                        vl_msk.transpose((1, 2, 0)) * 255)


def save_partial_conv_dataset(image_path, label_path, save_dir, split=0.6):
    mean, std = 9.2158, 1.5720
    min_, max_ = -3 * std, 3 * std
    im_transform = PairCompose([[Standardize(mean=mean, std=std), MaskClassReduction([0, 1, 2, 3], [0, 1], 0)],
                                [Normalize((0, 255), (min_, max_)), None],
                                [ToGrayscale(output_channels=3), ToGrayscale(output_channels=3)]])

    dataset = PMSE_Dataset(image_path, label_path, im_transform)

    folders = ['Train', 'Validation']
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

    for i in range(len(dataset)):
        im, msk, info = dataset[i]
        msk = msk.repeat(3, 1, 1)

        im = np.array(im)
        msk = np.array(msk)

        im = im.transpose((1, 2, 0))
        msk = msk.transpose((1, 2, 0))

        split_index = int(im.shape[-2] * split)

        train_im, val_im = im[:, 0:split_index, :], im[:, split_index:, :]
        train_msk, val_msk = msk[:, 0:split_index, :], msk[:, split_index:, :]

        cv2.imwrite(save_dir + '/train' + '/data' + '/' + info['image_name'] + '.png', train_im)
        cv2.imwrite(save_dir + '/train' + '/label' + '/' + info['image_name'] + '.png', train_msk)

        cv2.imwrite(save_dir + '/validation' + '/data' + '/' + info['image_name'] + '.png', val_im)
        cv2.imwrite(save_dir + '/validation' + '/label' + '/' + info['image_name'] + '.png', val_msk)

