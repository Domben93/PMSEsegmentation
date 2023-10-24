import numpy as np
import torch
import torch.nn
import argparse
from models.utils import load_model
from utils.dataset import get_dataloader
from utils import transforms as t
from utils.utils import *
from utils.metrics import SegMets


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
            f'../PMSE-segmentation/weights/Unet_plusspluss_64_pretrain-False_loss-DiceBCELoss_optim-adam_generated_dataset_shift_Hflip-Cadj_/lr_0.0008_wd_0.001_betas_0.9-0.999_momentum_0.9_freezed-None_{str(i)}.pt')
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

    #display(best_images[0], best_images[1], best_images[2], best_images[3], sample_classes)
    #save_images(best_images[0], best_images[1], best_images[2],
    #            save_path=f'C:\\Users\\dombe\\PycharmProjects\\PMSE-segmentation\\results\\images\\Unetpluss64-gendata-aug',
    #            names=best_images[3])


if __name__ == '__main__':

    torch.manual_seed(666)
    parser = argparse.ArgumentParser(description='Testing model for PMSE signal segmentation')
    parser.add_argument('--config-path', type=str, default='config\\unet_config.ymal',
                        help='Path to confg.ymal file (Default unet_config.ymal)')

    main(parser.parse_args())
