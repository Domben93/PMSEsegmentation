import numpy as np
import os
import torch
import torch.nn
import argparse
from models.utils import load_model
from utils.dataset import PMSEDatasetInference
from torch.utils.data import DataLoader
from utils import transforms as t
from utils.utils import *
from utils.metrics import SegMets
from models.unets import UNet, UNet_unet, UNet_vgg, PSPUNet
from models.unet_plusspluss.unet_plusspluss import Generic_UNetPlusPlus


def main(args):
    print('Running Inference')
    print(f'Saving results to: "{args.save}"')

    if not os.path.isdir(args.images):
        raise FileNotFoundError(f'Could not locate path "{args.images}"')
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    extension = check_extension(args.images)
    if not extension[0]:
        raise ValueError(f'All files in folder path must have same extension and of either .mat, .png or .jpg')

    if extension[1] == '.mat':
        min_, max_ = -3 * args.mean_std[1], 3 * args.mean_std[1]
        compose = t.Compose([t.Standardize(mean=args.mean_std[0], std=args.mean_std[1]),
                             t.Normalize((0, 1), (min_, max_), return_type=torch.float32),
                             t.ToGrayscale(output_channels=3),
                             t.ConvertDtype(torch.float32),
                             t.QuasiResize(args.resize_shape, args.max_resize_scale)
                             ])
    else:
        compose = t.Compose([
            t.ConvertDtype(torch.float32),
            t.Normalize((0, 1), (0, 255), return_type=torch.float32),
            t.QuasiResize(args.resize_shape, args.max_resize_scale)
        ])

    dataloader = DataLoader(PMSEDatasetInference(image_dir=args.images,
                                                 transform=compose,
                                                 last_min_overlap=0),
                            batch_size=1,
                            num_workers=1)

    undo_scaling = t.UndoQuasiResize(t.QuasiResize([64, 64], 2))

    images, labels, pred, info_list = [], [], [], []
    print('Testing')

    if args.model.lower() == 'unet':
        model = UNet(3, 1, initial_features=args.model_initfeatures)
    elif args.model.lower() == 'unet_plusspluss':
        model = Generic_UNetPlusPlus(input_channels=3,
                                     base_num_features=args.model_initfeatures,
                                     num_classes=1,
                                     num_pool=4,
                                     convolutional_pooling=False,
                                     convolutional_upsampling=True,
                                     deep_supervision=args.avg_supervision,
                                     init_encoder=None,
                                     seg_output_use_bias=True)
    else:
        raise ValueError(f'Model name is not supported')

    device = (torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu'))
    pre_trained = torch.load(args.weights)
    model.load_state_dict(pre_trained['model_state'])
    model.to(device)
    model.eval()

    for data in dataloader:
        image, info = data

        image = image.to(device)
        with torch.no_grad():
            res = model(image)

            if isinstance(res, tuple):
                if args.avg_supervision:
                    res = (res[0] + res[1] + res[2] + res[3]) / len(res)
                else:
                    res = res[0]

            res = res.view(1, 64, 64)

            (lr, rr), (lc, rc) = info['split_info']

            result_original_size = undo_scaling(res.detach().cpu(), (rr - lr, rc - lc))

            info_1 = remove_from_dataname(info['image_name'])
            if args.prediction_threshold > .0:
                result_original_size = torch.where(result_original_size >= args.prediction_threshold, 1, 0).to(torch.float32)
            pred.append(result_original_size[0, :, :])
            info_list.append(info_1[0])

    if args.save:
        save_results(preds=pred,
                     save_path=args.save,
                     names=info_list)


if __name__ == '__main__':
    torch.manual_seed(666)
    parser = argparse.ArgumentParser(description='Inference PMSE segmentation')
    parser.add_argument('--images', type=str, default='../PMSE-segmentation/dataset/Complete/data/',
                        help='Path to image data (Default: ../dataset/Test/data)')
    parser.add_argument('--model', type=str, default='Unet_plusspluss', help='Type of model architecture')
    parser.add_argument('--model-initfeatures', type=int, default=64, help='Number of initial feature maps')
    parser.add_argument('--weights', type=str, default='../PMSE-segmentation/weights/best_model.pt',
                        help='Path to pre-trained model weights')
    parser.add_argument('--save', type=str, default='../PMSE-segmentation/results/images/inference/',
                        help='Results save path')
    parser.add_argument('--resize-shape', type=list, default=[64, 64], help='Resize all data to the specified shape.'
                                                                            'Note that the original size will be'
                                                                            'returned if results are to be saved.')
    parser.add_argument('--max-resize-scale', type=int, default=2, help='Max scaling of image during resize')
    parser.add_argument('--mean-std', type=list, default=[9.2158, 1.5720], help='Mean and std value of PMSE to '
                                                                                'standardize data')
    parser.add_argument('--avg-supervision', type=bool, default=False, help='Average output from all output nodes.'
                                                                            'Only works for UNet++ architecture')
    parser.add_argument('--device', type=int, default=0, help='Device (GPU or CPU). Will use CPU if GPU is not found.')
    parser.add_argument('--prediction-threshold', type=float, default=0.5, help='Threhold for the raw predictions.'
                                                                                'If set to 0 then he raw predictions'
                                                                                'will be saved')
    main(parser.parse_args())
