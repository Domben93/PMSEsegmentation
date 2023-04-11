import os
import yaml
from utils.splitdata import SplitSegData, StratifiedSegDataSplit
import torch
import utils.transforms as t


def split_dataset(seed=42):
    with open('dataset_config.ymal') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    split_info = config['data_split']

    sample_split = split_info['train_val_test_split'] if sum(split_info['train_val_test_split']) == 1 else None

    if not sample_split:
        raise ValueError('Sum of split must be equal to 1.0 (e.g, 0.6/0.2/0.2 = 1.0)')

    data_path = os.path.join(config['data_path']['path_complete'], 'data')
    label_path = os.path.join(config['data_path']['path_complete'], 'label')

    if split_info['transform']:
        mean, std = config['data_vars']['mean'], config['data_vars']['std']

        min_, max_ = -3 * std, 3 * std
        im_transform = t.Compose([t.Standardize(mean=mean, std=std),
                                  t.Normalize((0, 255), (min_, max_)),
                                    t.ToGrayscale(output_channels=3)])
    else:
        im_transform = None

    if split_info['mask_reduction']:
        msk_transform = t.Compose([t.MaskClassReduction([0, 1, 2, 3], [0, 1], 0),
                                   t.ToGrayscale(output_channels=3)])
    else:
        msk_transform = None

    if split_info['sample_size_split']:

        if split_info['stratified_split']:

            strat_split = StratifiedSegDataSplit(image_dir=data_path,
                                                 label_dir=label_path,
                                                 random_split=True,
                                                 random_seed=seed)

            strat_split.save(save_dir='../Test/partial_conv/data',
                             specific='all',
                             img_transforms=im_transform,
                             mask_transforms=msk_transform)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


if __name__ == '__main__':
    split_dataset()
