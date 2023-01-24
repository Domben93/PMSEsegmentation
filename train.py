import os.path

import torch
import torch.nn as nn
import torch.optim as optimize
import argparse
import datetime
import time
from utils import transforms as t
from utils.loss import WeightedSoftDiceLoss
from utils.dataset import PMSE_Dataset
from utils.utils import *
from utils.loss import BinaryDiceLoss
from torch.utils.data import DataLoader
from models.unets import *
from config.settings import NetworkSettings as Settings
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import mIoU
from tqdm import tqdm


def argparser():
    parser = argparse.ArgumentParser(description='Training model for segmentation of PMSE signal')
    parser.add_argument('--batch-size', type=int, default=8, help='training batch size input (default: 8)')
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='Weight decay value (L2 penalty) (default: 0.001)')
    parser.add_argument('--worker', type=int, default=8, help='Number of workers for data loading (default: 8)')
    parser.add_argument('--device', type=str, default='cuda:0', help='training device (default: cuda:0)')
    parser.add_argument('--datafolder', type=str, default='')
    parser.add_argument('--weights', type=str, default=None, help='Full path to weights that is loaded to model. '
                                                                  '(default is None such that the model has default'
                                                                  ' initiation of weights according to how Pytorch'
                                                                  ' initiates weights to each layer)')
    parser.add_argument('--freeze-blocks', type=list, default='all', help='Specify the blocks that you want to freeze'
                                                                          'the weights on e.g. [0, 1, 2, 3, 4, 5] freezes'
                                                                          'block 0-5.'
                                                                          'By default all blocks are frozen')
    parser.add_argument('--validation', type=bool, default=False, help='Training validation on or off')
    parser.add_argument('--validation_interval', type=int, default=1, help='How often to run validation when training')
    parser.add_argument('--save', type=bool, default=True,
                        help='Save activated (disable if you dont want to store the result)')
    parser.add_argument('--save-loc', type=str, default=os.path.join(os.getcwd(), 'weights'),
                                                        help='Save model to location (default is current'
                                                                          ' working directory)')
    return parser.parse_args()


def main(args):

    print(f'Initiating training of model')
    device = (torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu'))

    pmse_train_data = PMSE_Dataset(Settings.CompleteSet.DATA,
                                   Settings.CompleteSet.MASKS,
                                   transform=t.ToTensor(),
                                   disable_warnings=True)

    data_loader = DataLoader(pmse_train_data, batch_size=1)

    mean, std = dataset_mean_and_std(data_loader)

    min_ = - std * 3
    max_ = std * 3
    im_size = 64
    #if args.freeze_blocks == 'all':
    #    freeze_blocks = []
    #freeze_blocks = []

    pair_compose = t.PairCompose([
        [t.Standardize(mean=mean, std=std), t.MaskClassReduction([0, 1, 2, 3], [0, 1], 0)],
        [t.Normalize((0, 255), (min_, max_)), None],
        [t.ToTensor(), t.ToTensor(zero_one_range=False)],
        [t.QuasiResize([im_size, im_size], 2), t.QuasiResize([im_size, im_size], 2)],
        [t.ToGrayscale(output_channels=3), None],
        [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)]
    ])
    print(mean, std)
    pass
    pmse_train_data = PMSE_Dataset(Settings.CompleteSet.DATA,
                                   Settings.CompleteSet.MASKS,
                                   transform=pair_compose,
                                   square_split=True,
                                   percent_overlap=0.3)
    train_loader = DataLoader(pmse_train_data, batch_size=args.batch_size, num_workers=args.worker, shuffle=True)

    if args.validation:
        pmse_val_data = PMSE_Dataset(Settings.Validation.DATA,
                                     Settings.Validation.MASKS,
                                     transform=pair_compose,
                                     square_split=True,
                                     percent_overlap=.0)

        val_loader = DataLoader(pmse_val_data, batch_size=args.batch_size, num_workers=args.worker)
    else:
        val_loader = None

    unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                in_channels=3, out_channels=1, init_features=32, pretrained=True)

    model = PSPUNet(in_channels=3, out_channels=1, initial_features=32, pooling_size=(1, 2, 4, 6)).to(device=device)
    #model = UNet(in_channels=3, out_channels=1, initial_features=32)
    #model = UNet(in_channels=3, out_channels=1, initial_features=32).to(device=device)
    print(len(list(model.state_dict())))
    print(len(list(unet_model.state_dict())))
    model.init_weigths_by_layer(unet_model, [i for i in range(116)])
    model.freeze_blocks(freeze_blocks=[0, 1, 2, 3, 5, 6, 7, 8], freeze_layer_types=[nn.Conv2d])

    optimizer = optimize.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #loss_func = nn.BCELoss()
    loss_func = BinaryDiceLoss()

    writer_name = (f'{type(model).__name__}_{str(im_size)}x{str(im_size)}'
                   f'_lr{args.lr}_freezed01235678_diceloss_extratest')

    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), f'runs\\{type(model).__name__}', writer_name))
    writer.add_graph(model, input_to_model=torch.rand(1, 3, 64, 64).to(device))
    writer.add_text(tag='Run Info', text_string='Hey')
    if args.weights:
        model.init_weights(args.weights)

    model.to(device)

    iou_met = mIoU()

    loss_train = []
    iou_train = []

    loss_val = []
    iou_val = []

    best_model_state = model.state_dict()
    best_optim_state = optimizer.state_dict()
    best_met = 0
    epoch = 0

    print(f'Training on device: {device}')

    print(f'Starting Training of {type(model).__name__} with \n'
          f'--> Optimizer: {type(optimizer).__name__}\n'
          f'--> loss function: {type(loss_func).__name__}\n'
          f'--> learning rate: {args.lr}\n')

    for epoch in range(1, args.epochs + 1):

        t1 = time.time()
        model.train()
        train_loss, val_loss = .0, .0

        for num, data in enumerate(train_loader):
            x, mask, data_info = data
            x = x.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(mode=True):
                predicted = model(x)

                loss = loss_func(predicted, mask)

                iou_met(predicted, mask)

                train_loss += loss.item()

                loss.backward()
                optimizer.step()

        loss_train.append(train_loss / len(train_loader))
        iou_train.append(iou_met.compute())

        iou_met.reset()

        writer.add_scalar("Loss/train", loss_train[-1], epoch)
        writer.add_scalar("mIoU/train", iou_train[-1], epoch)

        if val_loader and (epoch % args.validation_interval == 0 or epoch == 1):
            model.eval()

            for data in val_loader:
                x, mask, data_info = data

                x = x.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    predicted = model(x)

                    val_loss = loss_func(predicted, mask)
                    iou_met(predicted, mask)

                    val_loss += val_loss.item()

            loss_val.append(val_loss / len(val_loader))
            iou_val.append(iou_met.compute())

            iou_met.reset()

            writer.add_scalar('Loss/validate', loss_val[-1], epoch)
            writer.add_scalar('mIoU/validate', iou_val[-1], epoch)

        if epoch == 1 or epoch % 1 == 0:
            print(f'{datetime.datetime.now()}, Epoch {epoch}:\n'
                  f'Training: loss {loss_train[-1]:.6f}, mIoU {iou_train[-1]:.6f}')
            if val_loader:
                print(f'Validation: loss {loss_val[-1]:.6f}, mIoU {iou_val[-1]:.6f}')

        if (epoch == 1 or loss_train[-1] < best_met) and args.save:
            best_met = loss_train[-1]
            best_model_state = model.state_dict()
            best_optim_state = optimizer.state_dict()

        t2 = time.time()

        print(f'Estimated time left: {(t2 - t1) * (args.epochs - epoch)}')

    if args.save:
        hist_dict = {'train': {
            'loss': loss_train,
            'mIoU': iou_train,
        },
            'validation': {
                'loss': loss_val,
                'mIoU': iou_val,
            }}

        save_model(model=model,
                   optimizer=loss_func,
                   epoch=epoch,
                   train_samples=len(pmse_train_data),
                   file_name=os.path.join(writer_name + '.pt'),
                   save_path=args.save_loc,
                   loss_history=hist_dict,
                   model_state_dict=best_model_state,
                   optimizer_state_dict=best_optim_state)

    writer.close()


if __name__ == '__main__':
    torch.manual_seed(666)
    main(argparser())
