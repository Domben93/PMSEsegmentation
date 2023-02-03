import numpy as np
import torch
import torch.nn
import argparse
from utils.dataset import PMSE_Dataset, get_dataloader
from utils import transforms as t
from torch.utils.data import DataLoader
from utils.utils import *
from utils.model_comparison import sde, cev
from matplotlib import pyplot as plt
from models import unets
from config.settings import NetworkSettings as Settings
from utils import metrics as met


def display(image, label, pred, info, v_min, v_max):
    if not len(image) == len(label) == len(pred):
        raise ValueError('List of images, labels and predictions must be of equal length')

    fig_list = []

    images = [d['image_name'] for d in info]
    sizes = [d['split_info'] for d in info]
    # print(images)
    # print(sizes)
    unique_images = np.unique(np.array(images))

    total_count = 0
    for image_name in unique_images:
        count = images.count(image_name)

        total_count += count

        fig, axes = plt.subplots(3, count, figsize=(16, 7))
        if len(axes.shape) == 1:
            axes = np.array([axes]).T
        fig.add_gridspec(3, count, wspace=0.0, hspace=0.0)

        string = ['Data', 'Label', 'Prediction']

        for m in range(3):
            for n in range(count):

                if n == 0:
                    axes[m, n].set_ylabel(string[m], rotation=90)
                if m == 0:
                    axes[m, n].set_title('Part ' + str(n + 1))

                if m == 0:
                    arr = image[total_count - count:total_count]
                    min_ = v_min
                    max_ = v_max
                elif m == 1:
                    arr = label[total_count - count:total_count]
                    min_ = 0
                    max_ = 1
                else:
                    arr = pred[total_count - count:total_count]
                    min_ = 0
                    max_ = 1

                if n + 1 == count:
                    (lr_0, rr_0), (lc_0, rc_0) = sizes[total_count - 2]
                    (lr_1, rr_1), (lc_1, rc_1) = sizes[total_count - 1]

                    diff_r = 0  # (rr_1 - lr_1) - ((rr_1 - lr_1) - (rr_0 - lr_1)) - Does not work the way intended
                    diff_c = (rc_1 - lc_1) - ((rc_1 - lc_1) - (rc_0 - lc_1))
                    arr[n] = arr[n][diff_r:, diff_c:]

                axes[m, n].imshow(arr[n], cmap='jet', vmin=min_, vmax=max_, aspect='equal')
                plt.setp(axes[m, n].get_xticklabels(), visible=False)
                plt.setp(axes[m, n].get_yticklabels(), visible=False)
                axes[m, n].tick_params(axis='both', which='both', length=0)
        fig.suptitle(image_name)

        fig_list.append(fig)

    return fig_list


def main(args):
    config = load_yaml_as_dotmap(args.config_path)
    print('Running test')

    pair_compose = t.PairCompose([
        [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)],
        [t.Normalize((0, 1), (0, 255), return_type=torch.float32), None],
        [t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale),
         t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale)],
    ])

    test_loader = get_dataloader(args.config_path, pair_compose, mode='test')

    model = unets.UNet(3, 1, 32)
    device = torch.device("cuda")
    pre_trained = torch.load('../Test/weights/UNet_Train_randominit_unet_BCE_sdg.pt')
    model.init_weights(pre_trained['model_state'])

    model.to(device)
    model.eval()

    images, labels, pred, info_list = [], [], [], []

    sample_classes = [['MAD6400_2008-07-02_arcd_60@vhf_400749', 'MAD6400_2009-06-10_arcd_60@vhf_422844',
                       'MAD6400_2009-07-16_manda_60@vhf_655441', 'MAD6400_2011-06-01_manda_59',
                       'MAD6400_2015-08-10_manda_59',
                       'MAD6400_2015-08-13_manda_59', 'MAD6400_2015-08-20_manda_59'],
                      ['MAD6400_2008-06-30_manda_60@vhf_060693', 'MAD6400_2009-07-14_manda_60@vhf_684778',
                       'MAD6400_2009-07-17_manda_60@vhf_633279', 'MAD6400_2009-07-30_manda_60@vhf_779294',
                       'MAD6400_2010-07-08_manda_59',
                       'MAD6400_2010-07-09_manda_60@vhf_470083', 'MAD6400_2011-06-08_manda_59',
                       'MAD6400_2011-06-09_manda_59',
                       'MAD6400_2014-07-01_manda_48@vhf_178333', 'MAD6400_2015-08-12_manda_59'],
                      ['MAD6400_2010-07-07_manda_60@vhf_576698']]

    image_names = [item for sublist in sample_classes for item in sublist]

    metrics = met.SegMets()
    baseline_metric = met.SegMets()

    undo_scaling = t.UndoQuasiResize(t.QuasiResize([64, 64], 2))

    for data in test_loader:
        image, mask, info = data

        image = image.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            res = model(image)

            res = res.view(1, 64, 64)
            image = image.view(3, 64, 64)
            mask = mask.view(1, 64, 64)

            (lr, rr), (lc, rc) = info['split_info']
            image_original_size = undo_scaling(image.detach().cpu(), [rr - lr, rc - lc])
            mask_original_size = undo_scaling(mask.detach().cpu(), [rr - lr, rc - lc])
            result_original_size = undo_scaling(res.detach().cpu(), [rr - lr, rc - lc])

            for i, name in enumerate(info['image_name']):

                info['image_name'][i] = info['image_name'][i].rpartition('_')[0].rpartition('_')[0]

            metrics(result_original_size, mask_original_size, info['image_name'])

            baseline_metric(torch.randint(0, 2, result_original_size.shape), mask_original_size, info['image_name'])

            images.append(image_original_size[0, :, :])
            labels.append(mask_original_size[0, :, :])
            # res = (res >= 0.5).float().detach().cpu().numpy()[0, :, :]
            res = (result_original_size >= 0.5).float().view(1, rr - lr, rc - lc)[0, :, :]
            pred.append(res)
            info_list.append(info)

    print(f'mIoU: {metrics.mIoU()}')
    #print(f'AUC: {metrics.auc()}')
    #print(f'Accuracy: {metrics.accuracy()}')
    #print(f'Precision: {metrics.precision()}')
    #print(f'Dice Coef: {metrics.dice()}')

    baseline_fnr_fpr = baseline_metric.FPR_FNR(sample_classes=sample_classes)
    alternative_fnr_fpr = metrics.FPR_FNR(sample_classes=sample_classes)

    print(baseline_fnr_fpr)
    print(alternative_fnr_fpr)

    print(sde(alternative_fnr_fpr, baseline_fnr_fpr))
    print(cev(alternative_fnr_fpr, baseline_fnr_fpr))
    # figs = display(images, labels, pred, info_list, 0, 1)

    # plt.show()


if __name__ == '__main__':
    torch.manual_seed(666)
    parser = argparse.ArgumentParser(description='Testing model for PMSE signal segmentation')
    parser.add_argument('--config-path', type=str, default='models\\options\\unet_config.ymal',
                        help='Path to confg.ymal file (Default unet_config.ymal)')

    main(parser.parse_args())
