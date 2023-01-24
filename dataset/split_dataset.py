import os
import yaml
from utils.splitdata2 import SplitSegData, StratifiedSegDataSplit
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
        mean, std = torch.tensor([config['data_vars']['mean']]), torch.tensor([config['data_vars']['std']])

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

            strat_split = StratifiedSegDataSplit(image_dir=os.path.join(config['data_path']['path_complete'], 'data'),
                                                 label_dir=os.path.join(config['data_path']['path_complete'], 'label'),
                                                 random_split=True,
                                                 random_seed=seed)

            strat_split.save(save_dir='../dataset',
                             specific='all',
                             img_transforms=im_transform,
                             mask_transforms=msk_transform)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    """
    split_data = SplitSegData(image_dir=data_path,
                              label_dir=label_path,
                              square_split=split_info['square_split'],
                              split_overlap=split_info['split_overlap'],
                              last_min_overlap=split_info['last_min_overlap'],
                              get_trailing=True)

    train_dict = OrderedDict()
    val_dict = OrderedDict()
    test_dict = OrderedDict()

    if split_info['sample_size_split']:

        if len(split_info['sample_size_range']) == 0:
            raise ValueError(f'Sample size range must be bigger than 0. i.e., at least one value must be given'
                             f'to separate the different samples by the number of sub-samples.')

        split_dict = {}.fromkeys([i for i in range(len(split_info['sample_size_range']) + 1)], None)

    else:
        split_dict = {}.fromkeys([i for i in range(len(os.listdir(data_path)))], None)

    for key in split_dict.keys():
        split_dict[key] = []

    store_samples = []
    key_iter = iter(list(split_dict.keys()))

    for sample in range(len(split_data)):

        image, label, info = split_data[sample]

        if sample != len(split_data) - 1:
            next_img_name = split_data[sample+1][2]['image_name']
        else:
            next_img_name = ''  # empty string for last case

        store_samples.append([image, label, info])

        if info['image_name'] != next_img_name:

            if split_info['sample_size_split']:

                for num, range_ in enumerate(split_info['sample_size_range'] + [numpy.Inf]):

                    if len(store_samples) < range_:
                        split_dict[num].append(store_samples)
                        store_samples = []
                        break
            else:
                key = next(key_iter)
                split_dict[key].append(store_samples)
                store_samples = []

    random.seed(split_info['seed'])

    for key in split_dict.keys():

        data_class = split_dict[key]
        #random.shuffle(data_class)
        len_sample_class = len(data_class)
        #print(data_class)
        if split_info['stratified_split'] and split_info['sample_size_split']:

            for sample in data_class:
                print(len(sample))
                if len_sample_class <= len(split_info['train_val_test_split']):
                    pass
                else:
                    pass
        elif split_info['stratified_split']:
            pass#print(len_sample_class)
    else:
        pass
"""


if __name__ == '__main__':
    split_dataset()
