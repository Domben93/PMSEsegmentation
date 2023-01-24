import numpy as np
import torch
import torch.nn
import argparse
from utils.dataset import PMSE_Dataset
from utils import transforms as t
from torch.utils.data import DataLoader
from utils.utils import *
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


def argparser():
    parser = argparse.ArgumentParser(description='Testing model for PMSE signal segmentation')
    parser.add_argument('--weights', type=str, )
    return parser.parse_args()


def main():
    args = argparser()

    print('Testing')

    pmse_train_data = PMSE_Dataset(Settings.CompleteSet.DATA,
                                   Settings.CompleteSet.MASKS,
                                   disable_warnings=True)

    data_loader = DataLoader(pmse_train_data, batch_size=1)

    mean, std = dataset_mean_and_std(data_loader)

    min_ = - std * 3
    max_ = std * 3

    v_min, v_max = 0, 1

    pair_compose = t.PairCompose([
        [t.Standardize(mean=mean, std=std), t.MaskClassReduction([0, 1, 2, 3], [0, 1], 0)],
        [t.Normalize((0, 255), (min_, max_)), None],
        [t.ToTensor(), t.ToTensor(zero_one_range=False)],
        [t.QuasiResize([64, 64], 2), t.QuasiResize([64, 64], 2)],
        [t.ToGrayscale(output_channels=3), None],
        [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)]
    ])

    pmse_test_data = PMSE_Dataset(Settings.IntermediateSamples.DATA,
                                  Settings.IntermediateSamples.MASKS,
                                  transform=pair_compose,
                                  square_split=True,
                                  percent_overlap=.0)

    pmse_test_class1 = PMSE_Dataset(Settings.BigSamples.DATA,
                                    Settings.BigSamples.MASKS,
                                    transform=pair_compose,
                                    square_split=True,
                                    percent_overlap=.0)

    pmse_test_class2 = PMSE_Dataset(Settings.IntermediateSamples.DATA,
                                    Settings.IntermediateSamples.MASKS,
                                    transform=pair_compose,
                                    square_split=True,
                                    percent_overlap=.0)

    pmse_test_class3 = PMSE_Dataset(Settings.SmallSamples.DATA,
                                    Settings.SmallSamples.MASKS,
                                    transform=pair_compose,
                                    square_split=True,
                                    percent_overlap=.0)

    class_list = [pmse_test_class1, pmse_test_class2, pmse_test_class3]

    UNet = unets.UNet(3, 1, 32)
    device = torch.device("cuda")
    model = torch.load('weights\\UNet_64x64_lr0.001_freezed01235678_diceloss.pt')
    UNet.init_weights(model['model_state'])

    UNet.to(device)
    UNet.eval()

    images, labels, pred, info_list = [], [], [], []
    miou = met.mIoU(threshold=0.5)
    bin_metrics = met.BinaryMetrics()
    # softDice = met.DiceCoefficient()
    dice = 0

    undo_scaling = t.UndoQuasiResize(t.QuasiResize([64, 64], 2))

    for i in range(len(pmse_test_data)):
        image, mask, info = pmse_test_data[i]

        image = image.to(device)[None, :]
        mask = mask.to(device)

        with torch.no_grad():
            res = UNet(image)

            res = res.view(1, 64, 64)
            image = image.view(3, 64, 64)
            mask = mask.view(1, 64, 64)

            miou(res, mask)
            # diceCoef = softDice(res, mask)

            bin_metrics(res.detach().cpu(), mask.detach().cpu())
            # dice += diceCoef.item()

            # o_image, o_mask, _ = pmse_test_data.get_original_image_and_mask(i)

            (lr, rr), (lc, rc) = info['split_info']
            image_original_size = undo_scaling(image.detach().cpu(), [rr - lr, rc - lc])
            mask_original_size = undo_scaling(mask.detach().cpu(), [rr - lr, rc - lc])
            result_original_size = undo_scaling(res.detach().cpu(), [rr - lr, rc - lc])

            images.append(image_original_size[0, :, :])
            labels.append(mask_original_size[0, :, :])
            # res = (res >= 0.5).float().detach().cpu().numpy()[0, :, :]
            res = (result_original_size >= 0.5).float().view(1, rr - lr, rc - lc)[0, :, :]
            pred.append(res)
            info_list.append(info)

    # print(f'AUC: {auc.auc()}')
    # print(f'Dice: {dice / len(pmse_test_data)}')
    print(f'mIoU: {miou.compute()}')
    bin_metrics.confusion_matrix(threshold=0.3)
    print(bin_metrics.FNR(), bin_metrics.FPR())
    print(f'Accuracy: {bin_metrics.accuracy()}')
    print(f'mIoU: {bin_metrics.mIoU()}')
    figs = display(images, labels, pred, info_list, v_min, v_max)

    plt.show()


if __name__ == '__main__':
    main()
