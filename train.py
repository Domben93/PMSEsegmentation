import importlib
import os.path
import torch
import argparse
import datetime
import time

import utils.loss
from utils import transforms as t
from utils.dataset import get_dataloader
from utils.utils import *
from models.utils import *
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import mIoU


def main(args):

    print(f'Preparing training of model')

    config = load_yaml_as_dotmap(args.config_path)

    device = (torch.device(config.gpu) if torch.cuda.is_available() else torch.device('cpu'))

    augmentation_list = [t.RandomHorizontalFlip(0.25),
                         t.RandomVerticalFlip(0.25)]

    train_pair_compose = t.PairCompose([
        [t.ObjectAugmentation(augmentation_list)],
        [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)],
        [t.Normalize((0, 1), (0, 255), return_type=torch.float32), None],
        [t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale),
         t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale)],
    ])

    val_pair_compose = t.PairCompose([
        [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)],
        [t.Normalize((0, 1), (0, 255), return_type=torch.float32), None],
        [t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale),
         t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale)],
    ])

    train_loader = get_dataloader(args.config_path, train_pair_compose)

    if config.validation.validation_interval is not None:
        validate_loader = get_dataloader(args.config_path, val_pair_compose, mode='validate')
    else:
        validate_loader = None

    #unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #                            in_channels=3, out_channels=1, init_features=32, pretrained=True)

    model = load_model(args.config_path)
    model.to(device)
    optimizer, lr_scheduler = load_optimizer(config, model, grad_true_only=True)

    loss_func = getattr(importlib.import_module('utils.loss'), config.optimizer.loss_type)()
    #loss_func = utils.loss.BinaryDiceLoss()
    writer_name = f'{type(model).__name__}_Test1'

    writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), f'runs\\{type(model).__name__}', writer_name))
    writer.add_graph(model, input_to_model=torch.rand(1, 3, 64, 64).to(device))
    writer.add_text(tag='Run Info', text_string='Hey')

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
          f'--> learning rate: {config.optimizer.learning_rate}\n')

    print(f'Training for {config.training.epochs} epochs')

    for epoch in range(1, config.training.epochs + 1):

        t1 = time.time()
        model.train()
        train_loss, val_loss = .0, .0

        for num, data in enumerate(train_loader):

            x, mask, info = data
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
                lr_scheduler.step()

        loss_train.append(train_loss / len(train_loader))
        iou_train.append(iou_met.compute())

        iou_met.reset()

        writer.add_scalar("Loss/train", loss_train[-1], epoch)
        writer.add_scalar("mIoU/train", iou_train[-1], epoch)

        if validate_loader and (epoch % config.training.eval_interval == 0 or epoch == 1):
            model.eval()

            for data in validate_loader:

                x, mask, data_info = data
                x = x.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    predicted = model(x)

                    val_loss = loss_func(predicted, mask)
                    iou_met(predicted, mask)

                    val_loss += val_loss.item()

            loss_val.append(val_loss / len(validate_loader))
            iou_val.append(iou_met.compute())

            iou_met.reset()

            writer.add_scalar('Loss/validate', loss_val[-1], epoch)
            writer.add_scalar('mIoU/validate', iou_val[-1], epoch)

        if epoch == 1 or epoch % 1 == 0:
            print(f'{datetime.datetime.now()}, Epoch {epoch}:\n'
                  f'Training: loss {loss_train[-1]:.6f}, mIoU {iou_train[-1]:.6f}')
            if validate_loader:
                print(f'Validation: loss {loss_val[-1]:.6f}, mIoU {iou_val[-1]:.6f}')

        if (epoch == 1 or loss_train[-1] < best_met) and config.training.save_best:
            best_met = loss_train[-1]
            best_model_state = model.state_dict()
            best_optim_state = optimizer.state_dict()

        t2 = time.time()

        print(f'Estimated time left: {(t2 - t1) * (config.training.epochs - epoch)}')

    if config.training.save_best:
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
                   train_samples=-1,
                   file_name=os.path.join(writer_name + '.pt'),
                   save_path=config.model.save_path,
                   loss_history=hist_dict,
                   model_state_dict=best_model_state,
                   optimizer_state_dict=best_optim_state)

    writer.close()


if __name__ == '__main__':
    torch.manual_seed(666)

    parser = argparse.ArgumentParser(description='Training model for segmentation of PMSE signal')

    parser.add_argument('--config-path', type=str, default='models\\options\\unet_config.ymal',
                        help='Path to confg.ymal file (Default unet_config.ymal)')

    """
    unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                in_channels=3, out_channels=1, init_features=32, pretrained=True)
    
    from models.unets import UNet
    unet = UNet(3, 1, 32)

    unet.init_weights(unet_model.state_dict())
    torch.save({'model_state': unet.state_dict()}, 'weights\\mateuszbuda-brain-segmentation-pytorch.pt')
    """

    main(parser.parse_args())

