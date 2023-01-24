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
from utils.model_comparison import CEV, SDE


def display(image, label, pred, info, v_min, v_max, performance):
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
                                   transform=t.ToTensor(),
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

    pmse_class1 = PMSE_Dataset(Settings.BigSamples.DATA,
                                    Settings.BigSamples.MASKS,
                                    transform=pair_compose,
                                    square_split=True,
                                    percent_overlap=.0)

    pmse_class2 = PMSE_Dataset(Settings.IntermediateSamples.DATA,
                                    Settings.IntermediateSamples.MASKS,
                                    transform=pair_compose,
                                    square_split=True,
                                    percent_overlap=.0)
    pmse_class3 = PMSE_Dataset(Settings.SmallSamples.DATA,
                                    Settings.SmallSamples.MASKS,
                                    transform=pair_compose,
                                    square_split=True,
                                    percent_overlap=.0)

    class_list = [pmse_class1, pmse_class2, pmse_class3]

    models = [[unets.UNet(3, 1, 32), 'weights\\UNet_64x64_lr0.001_freezed01235678_alldata_for_analysis.pt'],
              [unets.UNet(3, 1, 32), 'weights\\UNet_64x64_lr0.001_freezed01235678_diceloss.pt'],
              [unets.PSPUNet(3, 1, 32, pooling_size=(1, 2, 4, 6)), 'weights\\PSPUNet_64x64_lr0.001_freezed01235678_diceloss.pt'],
              [unets.PSPUNet(3, 1, 32, pooling_size=(1, 2, 4, 6)), 'weights\\PSPUNet_64x64_lr0.001_freezed01235678_diceloss_BigSamples.pt']]

    device = torch.device("cuda")
    """
    UNet = unets.UNet(3, 1, 32)
    model = torch.load('weights\\UNet_64x64_lr0.001_freezed01235678_diceloss.pt')
    UNet.init_weights(model['model_state'])

    UNet.to(device)
    UNet.eval()
    """
    mIoU = met.mIoU(threshold=0.5)
    bin_metrics = met.BinaryMetrics()

    undo_scaling = t.UndoQuasiResize(t.QuasiResize([64, 64], 2))
    fpr_fnr_arr = np.zeros((len(models), 2, len(class_list)))
    eval_metrics = np.zeros((len(models), 2, len(class_list)))
    for m, (model, weights_path) in enumerate(models):
        weights = torch.load(weights_path)
        model.init_weights(weights['model_state'])
        model.to(device)
        model.eval()

        for n, sample_class in enumerate(class_list):
            images, labels, pred, info_list = [], [], [], []

            for i in range(len(sample_class)):
                image, mask, info = sample_class[i]

                image = image.to(device)[None, :]
                mask = mask.to(device)

                with torch.no_grad():
                    res = model(image)

                    res = res.view(1, 64, 64)
                    image = image.view(3, 64, 64)
                    mask = mask.view(1, 64, 64)

                    #miou(res, mask)
                    # diceCoef = softDice(res, mask)
                    mIoU(res, mask)
                    bin_metrics(res.detach().cpu(), mask.detach().cpu())
                    # dice += diceCoef.item()

                    # o_image, o_mask, _ = pmse_test_data.get_original_image_and_mask(i)

                    """ 
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
                    """

            bin_metrics.confusion_matrix()
            fpr_fnr_arr[m, 0, n] = bin_metrics.FPR()
            fpr_fnr_arr[m, 1, n] = bin_metrics.FNR()
            acc = bin_metrics.accuracy()
            #miou = bin_metrics.mIoU()
            miou = mIoU.compute()
            mIoU.reset()
            eval_metrics[m, 0, n] = acc
            eval_metrics[m, 1, n] = miou
            bin_metrics.confusion_matrix(threshold=0.5)
            print(f'Model {m + 1}: {type(model).__name__} class {n + 1} Accuracy: {acc} mIoU: {miou}')
            bin_metrics.reset()

        del model
        #figs = display(images, labels, pred, info_list, v_min, v_max, performance=0)
        #plt.show()

    print(eval_metrics)
    num_models = len(models)
    CEV_corr = np.zeros((num_models, num_models))
    SDE_corr = np.zeros((num_models, num_models))
    sde = SDE()
    cev = CEV()
    #acc_corr = np.zeros((num_models, num_models))
    #miou_corr = np.zeros((num_models, num_models))

    for i in range(num_models):
        for j in range(num_models):
            SDE_corr[i, j] = sde(fpr_fnr_arr[i, :, :], fpr_fnr_arr[j, :, :])
            CEV_corr[i, j] = cev(fpr_fnr_arr[i, :, :], fpr_fnr_arr[j, :, :])
            #acc_corr[i, j] = (eval_metrics[i, 0])

    alpha = ['1', '2', '3', '4']

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(221)
    cax = ax.matshow(CEV_corr, interpolation='nearest')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + alpha)
    ax.set_yticklabels([''] + alpha)
    ax.set_title("CEV")

    ax = fig.add_subplot(222)
    cax = ax.matshow(SDE_corr, interpolation='nearest')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + alpha)
    ax.set_yticklabels([''] + alpha)
    ax.set_title('SDE')

    classes = ['Big', 'Intermediate', 'Small']
    unet_1_acc = eval_metrics[0, 0, :]
    unet_1_miou = eval_metrics[0, 1, :]

    unet_2_acc = eval_metrics[1, 0, :]
    unet_2_miou = eval_metrics[1, 1, :]

    unet_3_acc = eval_metrics[2, 0, :]
    unet_3_miou = eval_metrics[2, 1, :]

    unet_4_acc = eval_metrics[3, 0, :]
    unet_4_miou = eval_metrics[3, 1, :]

    barwidth = 0.20

    br1 = np.arange(len(unet_1_acc))
    br2 = [x + barwidth for x in br1]
    br3 = [x + barwidth for x in br2]
    br4 = [x + barwidth for x in br3]

    ax = fig.add_subplot(223)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel('Sample Class')
    ax.bar(br1, unet_1_acc, color='r', width=barwidth, edgecolor='grey', label='Model 1')
    ax.bar(br2, unet_2_acc, color='g', width=barwidth, edgecolor='grey', label='Model 2')
    ax.bar(br3, unet_3_acc, color='b', width=barwidth, edgecolor='grey', label='Model 3')
    ax.bar(br4, unet_4_acc, color='y', width=barwidth, edgecolor='grey', label='Model 4')
    ax.set_xticklabels([""] + classes)

    ax = fig.add_subplot(224)

    ax.set_ylabel("mIoU (%)")
    ax.set_xlabel('Sample Class')
    ax.bar(br1, unet_1_miou, color='r', width=barwidth, edgecolor='grey', label='Model 1')
    ax.bar(br2, unet_2_miou, color='g', width=barwidth, edgecolor='grey', label='Model 2')
    ax.bar(br3, unet_3_miou, color='b', width=barwidth, edgecolor='grey', label='Model 3')
    ax.bar(br4, unet_4_miou, color='y', width=barwidth, edgecolor='grey', label='Model 4')
    ax.set_xticklabels([""] + classes)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


if __name__ == '__main__':
    main()
