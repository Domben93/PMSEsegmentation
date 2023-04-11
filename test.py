import os.path

import numpy as np
import torch
import torch.nn
import argparse
from models.utils import load_model
from utils.dataset import get_dataloader
from utils import transforms as t
from utils.utils import *
from matplotlib import pyplot as plt
from utils.metrics import SegMets, sde, cev
from torchvision.utils import save_image


def save_images(images, labels, preds, save_path, names):
    if not len(images) == len(labels) == len(preds):
        raise ValueError('List of images, labels and predictions must be of equal length')

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        os.mkdir(os.path.join(save_path, 'images'))
        os.mkdir(os.path.join(save_path, 'labels'))
        os.mkdir(os.path.join(save_path, 'preds'))

    for info in names:

        name_indices = [i for i, j in enumerate(names) if j == info]

        if len(name_indices) > 1:
            im = torch.cat([images[index] for index in name_indices], dim=1)
            label = torch.cat([labels[index] for index in name_indices], dim=1)
            pred = torch.cat([preds[index] for index in name_indices], dim=1)
        else:
            im = images[name_indices[0]]
            label = labels[name_indices[0]]
            pred = preds[name_indices[0]]

        save_image(im, save_path + '/images/' + f'im_{info}' + '.png')
        save_image(label, save_path + '/labels/' + f'mask_{info}' + '.png')
        save_image(pred, save_path + '/preds/' + f'label_{info}' + '.png')


def display(images, labels, preds, info, sample_classes):
    if not len(images) == len(labels) == len(preds):
        raise ValueError('List of images, labels and predictions must be of equal length')

    for sample_class in sample_classes:

        fig, ax = plt.subplots(4, len(sample_class))
        fig.add_gridspec(4, len(sample_class), wspace=0.0, hspace=0.0)

        for n, name in enumerate(sample_class):

            name_indices = [i for i, j in enumerate(info) if j == name]

            if len(name_indices) > 1:
                im = torch.cat([images[index] for index in name_indices], dim=1)
                label = torch.cat([labels[index] for index in name_indices], dim=1)
                pred = torch.cat([preds[index] for index in name_indices], dim=1)
            else:
                im = images[name_indices[0]]
                label = labels[name_indices[0]]
                pred = preds[name_indices[0]]

            ax[0, n].imshow(im[:, :], cmap='jet', vmin=0, vmax=1)
            ax[1, n].imshow(label[:, :], cmap='jet', vmin=0, vmax=1)
            ax[2, n].imshow(pred[:, :], cmap='jet', vmin=0, vmax=1)
            ax[3, n].imshow(np.where(pred[:, :] >= 0.5, 1, 0), cmap='jet', vmin=0, vmax=1)
            ax[0, n].axis('off')
            ax[1, n].axis('off')
            ax[2, n].axis('off')
            ax[3, n].axis('off')
        plt.show()

        """
        total_count += 1


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
        """


def main(args):
    config = load_yaml_as_dotmap(args.config_path)
    print('Running test')

    pair_compose = t.PairCompose([
        [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)],
        [t.Normalize((0, 1), (0, 255), return_type=torch.float32), None],
        [t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale),
         t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale)]
    ])

    test_loader = get_dataloader(args.config_path, pair_compose, mode='test')

    sample_classes = config.sample_classes.class_granular
    #sample_classes =[['img1', 'MAD6400_2014-07-01_manda_48@vhf_178333']]
    metrics = SegMets()

    all_miou = []
    all_dsc = []
    all_iou_classwise = []
    all_dsc_classwise = []

    best_iou = 0

    best_images = [[], [], [], []]
    undo_scaling = t.UndoQuasiResize(t.QuasiResize([64, 64], 2))
    for i in range(5):

        images, labels, pred, info_list = [], [], [], []
        print('Testing')
        model = load_model(args.config_path)
        device = torch.device("cuda")
        pre_trained = torch.load(
            f'../Test/weights/Unet_plusspluss_64_pretrain-False_loss-DiceBCELoss_optim-adam_generated_dataset_random-erase_Hflip-Cadj_/lr_0.0005_wd_0.001_betas_0.9-0.999_momentum_0.9_freezed-None_{str(i)}.pt')
        model.load_state_dict(pre_trained['model_state'])
        model.to(device)
        model.eval()

        for data in test_loader:
            image, mask, info = data

            image = image.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                res = model(image)

                if isinstance(res, tuple):
                    if config.model_init.deep_supervision:
                        res = (res[0] + res[1] + res[2] + res[3]) / len(res)
                    else:
                        res = res[0]

                res = res.view(1, 64, 64)
                image = image.view(3, 64, 64)
                mask = mask.view(1, 64, 64)

                (lr, rr), (lc, rc) = info['split_info']

                image_original_size = undo_scaling(image.detach().cpu(), (rr - lr, rc - lc))
                mask_original_size = undo_scaling(mask.detach().cpu(), (rr - lr, rc - lc))
                result_original_size = undo_scaling(res.detach().cpu(), (rr - lr, rc - lc))

                info_1 = remove_from_dataname(info['image_name'])

                metrics(result_original_size, mask_original_size, info_1)
                images.append(image_original_size[0, :, :])
                labels.append(mask_original_size[0, :, :])
                pred.append(result_original_size[0, :, :])
                info_list.append(info_1[0])

        miou = metrics.mIoU(sample_classes=sample_classes, multidim_average="global")
        dsc = metrics.dice(sample_classes=sample_classes, multidim_average='global')
        miou_classwise = metrics.mIoU(sample_classes=sample_classes, multidim_average="classwise")
        dsc_classwise = metrics.dice(sample_classes=sample_classes, multidim_average='classwise')
        metrics.reset()

        all_miou.append(miou)
        all_dsc.append(dsc)
        all_iou_classwise.append(miou_classwise)
        all_dsc_classwise.append(dsc_classwise)

        if miou > best_iou:
            best_iou = miou
            best_images[0], best_images[1], best_images[2], best_images[3] = images, labels, pred, info_list

    print(f'mIoU - mean: {np.mean(all_miou)}, std: {np.std(all_miou)}, all: {all_miou}')
    print(f'DSC -  mean: {np.mean(all_dsc)}, std: {np.std(all_dsc)}, all: {all_dsc}')
    print(all_miou)
    print(np.mean(all_iou_classwise, axis=0), np.std(all_iou_classwise, axis=0))
    print(np.mean(all_dsc_classwise, axis=0), np.std(all_dsc_classwise, axis=0))
    """
    fig, ax = plt.subplots(2, 3)
    fig.suptitle('Unet64-randominit - Original Dataset')
    ax[0, 0].imshow(best_images[0][0], cmap='jet', vmin=0, vmax=1)
    ax[0, 0].set_title('Image')
    ax[0, 0].set_ylabel('Removed easy')
    ax[0, 1].imshow(best_images[1][0], vmin=0, vmax=1, cmap='jet')
    ax[0, 1].set_title('GT')
    ax[0, 2].imshow(torch.where(best_images[2][0] >= 0.5, 1, 0), vmin=0, vmax=1, cmap='jet')
    ax[0, 2].set_title('Predicted')
    ax[1, 0].imshow(best_images[0][1], cmap='jet', vmin=0, vmax=1)
    ax[1, 0].set_ylabel('Original')
    ax[1, 1].imshow(best_images[1][1], vmin=0, vmax=1, cmap='jet')
    ax[1, 2].imshow(torch.where(best_images[2][1] >= 0.5, 1, 0), vmin=0, vmax=1, cmap='jet')
    for i in range(2):
        for j in range(3):
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.savefig('unet64-randominit-orgdata-easyremoved', dpi=400, bbox_inches='tight')
    plt.show()
    """

    #display(best_images[0], best_images[1], best_images[2], best_images[3], sample_classes)
    #save_images(best_images[0], best_images[1], best_images[2],
    #            save_path=f'C:\\Users\\dombe\\PycharmProjects\\Test\\results\\images\\Unetpluss64-gendata-aug',
    #            names=best_images[3])


if __name__ == '__main__':

    torch.manual_seed(666)
    parser = argparse.ArgumentParser(description='Testing model for PMSE signal segmentation')
    parser.add_argument('--config-path', type=str, default='models\\options\\unet_config.ymal',
                        help='Path to confg.ymal file (Default unet_config.ymal)')

    main(parser.parse_args())
