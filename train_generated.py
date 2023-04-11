import importlib
import math
import os.path
import random

import torch
import argparse
import datetime
import time
import utils.loss
from datetime import datetime, timedelta
from utils import transforms as t
from utils.dataset import get_dataloader
from utils.utils import *
from models.utils import *
from torch.utils.tensorboard import SummaryWriter
from utils.metrics import SegMets, sde, cev
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.functional.classification import binary_jaccard_index
from copy import deepcopy


def main(args):
    print(f'Preparing training of model')

    config = load_yaml_as_dotmap(args.config_path)

    device = (torch.device(config.gpu) if torch.cuda.is_available() else torch.device('cpu'))

    augmentation_list = [t.RandomHorizontalFlip(0.25),
                         t.RandomVerticalFlip(0.25)]

    train_pair_compose = t.PairCompose([
        [t.RandomHorizontalFlip(0.5)],
        # [t.RandomVerticalFlip(0.25)],
        [t.RandomContrastAdjust(0.5, (0.8, 1.2))],
        # [t.RandomBrightnessAdjust(0.5, (0.5, 1.5))],
        # [t.ObjectAugmentation(augmentation_list)],
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

    metrics = SegMets()
    best_model_state = {}
    best_optim_state = {}

    print(f'Training on device: {device}')

    pd_data = {
        'epoch': [],
        'run': [],
        'iou': [],
        'dsc': [],
        'loss': [],
        'event': []
    }

    sample_classes = config.sample_classes.class_names
    search_res = {}
    ds = '_DS' if config.model_init.deep_supervision else ''
    aug = 'Hflip-Cadj_'
    # {bool(config.model_init.pre_trained_weights)}_'
    writer_name = f'{str(config.model.model_type)}_{str(config.model_init.init_features)}_' \
                  f'pretrain-{bool(config.model_init.pre_trained_weights)}_' \
                  f'loss-{str(config.optimizer.loss_type)}_' \
                  f'optim-{str(config.optimizer.optim_type)}_' \
                  f'generated_dataset_random-erase_' + aug + ds

    file_name = f'lr_{config.optimizer.learning_rate}_' \
                f'wd_{config.optimizer.weight_decay}_' \
                f'betas_{config.optimizer.betas[0]}-{config.optimizer.betas[1]}_' \
                f'momentum_{config.optimizer.momentum}_' \
                f'freezed-{"".join([str(x) for x in config.model_init.freeze_layers]) if config.model_init.freeze_layers is not None else None}_'

    if not os.path.exists(os.path.join(config.model.save_path, writer_name)):
        os.mkdir(os.path.join(config.model.save_path, writer_name))

    if config.training.number_of_runs < 1:
        raise ValueError(f'Number of runs must be bigger than 1. Got {config.training.number_of_runs}.')

    for i in range(config.training.number_of_runs):

        best_met = 0 #math.inf
        model = load_model(args.config_path)
        model.to(device)
        optimizer, lr_scheduler = load_optimizer(config, model, grad_true_only=True,
                                                 lr=config.optimizer.learning_rate,
                                                 weight_decay=config.optimizer.weight_decay,
                                                 betas=config.optimizer.betas,
                                                 momentum=config.optimizer.momentum)

        loss_func = getattr(importlib.import_module('utils.loss'), config.optimizer.loss_type)() #utils.loss.DiceSurfaceLoss(strategy='increase', inc_reb_rate_iter=10)

        print(f'Starting Training of {type(model).__name__} with \n'
              f'--> Optimizer: {type(optimizer).__name__}\n'
              f'--> Optimizer Params: {optimizer.defaults}'
              f'--> loss function: {type(loss_func).__name__}\n'
              f'--> learning schedule: Step size: {config.learning_scheduler.step_size},'
              f' gamma: {config.learning_scheduler.gamma}, last epoch: {config.learning_scheduler.last_epoch} \n',
              f'--> Number of same model run: {i + 1} of {config.training.number_of_runs}')

        search_res[i] = {
            'miou': [],
            'dsc': [],
            'auc': [],
            'acc': [],
        }

        epoch = 0
        if config.training.early_stopping.early_stop:
            early_stopping = EarlyStopper(patience=config.training.early_stopping.patience,
                                          min_delta=config.training.early_stopping.min_delta)
        else:
            early_stopping = False
        iter_ = 0
        running = True
        while running:

            model.train()
            train_loss = .0

            for num, data in enumerate(train_loader):
                x, mask, info = data
                x = x.to(device)
                mask = mask.to(device)
                optimizer.zero_grad()
                info = remove_from_dataname_extended(info['image_name'], sample_classes)

                with torch.set_grad_enabled(mode=True):
                    predicted = model(x)

                    if config.model_init.deep_supervision:
                        losses = []
                        for pred in predicted:
                            losses.append(loss_func(pred, mask))
                        losses.reverse()

                        for l in losses:
                            if l == losses[-1]:
                                l.backward()
                                optimizer.step()
                            else:
                                l.backward(retain_graph=True)

                        if config.model_init.avg_output:
                            predicted = (predicted[0] + predicted[1] + predicted[2] + predicted[3]) / len(predicted)
                        else:
                            predicted = predicted[0]

                        metrics(predicted.detach().cpu(), mask.detach().cpu(), info)
                        train_loss += ((losses[0].item() + losses[1].item() + losses[2].item() + losses[3].item()) / len(losses))
                    else:

                        loss = loss_func(predicted, mask)

                        metrics(predicted.detach().cpu(), mask.detach().cpu(), info)

                        train_loss += loss.item()

                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()

                if iter_ % config.training.eval_interval == 0:

                    loss_train = (float(train_loss / (num + 1)))
                    miou_train = float(metrics.mIoU(sample_classes=sample_classes, multidim_average='global'))
                    dsc_train = float(metrics.dice(sample_classes=sample_classes, multidim_average='global'))

                    metrics.reset()

                    pd_data['epoch'].append(iter_)
                    pd_data['iou'].append(miou_train)
                    pd_data['dsc'].append(dsc_train)
                    pd_data['loss'].append(loss_train)
                    pd_data['event'].append('Train')
                    pd_data['run'].append(i)

                    model.eval()
                    val_loss = .0
                    for data in validate_loader:
                        x, mask, info = data
                        x = x.to(device)
                        mask = mask.to(device)
                        with torch.no_grad():
                            predicted = model(x)

                            if isinstance(predicted, tuple):
                                if config.model_init.deep_supervision:
                                    predicted = (predicted[0] + predicted[1] + predicted[2] + predicted[3]) / len(
                                        predicted)
                                else:
                                    predicted = predicted[0]

                            loss = loss_func(predicted, mask)
                            info = remove_from_dataname(info['image_name'])

                            metrics(predicted.detach().cpu(), mask.detach().cpu(), info)

                            val_loss += loss.item()

                    loss_val = (float(val_loss / len(validate_loader)))
                    miou_val = float(metrics.mIoU(sample_classes=sample_classes, multidim_average='global'))
                    dsc_val = float(metrics.dice(sample_classes=sample_classes, multidim_average='global'))

                    metrics.reset()
                    model.train()

                    pd_data['epoch'].append(iter_)
                    pd_data['iou'].append(miou_val)
                    pd_data['loss'].append(loss_val)
                    pd_data['dsc'].append(dsc_val)
                    pd_data['event'].append('Validation')
                    pd_data['run'].append(i)

                    print(f'{datetime.now()}, Iter {iter_} of {config.training.iters}:\n'
                          f'Training: loss {loss_train:.6f},'
                          f' mIoU {miou_train:.6f},'
                          f' dsc {dsc_train:.6f}')
                    print(f'Validation: loss {loss_val:.6f},'
                          f' mIoU {miou_val:.6f},'
                          f' dsc {dsc_val:.6f}')

                    if dsc_val > best_met and config.training.save_best:
                        best_met = dsc_val
                        best_model_state = deepcopy(model.state_dict())
                        best_optim_state = deepcopy(optimizer.state_dict())
                        print('Saving best model')

                    if iter_ == config.training.iters:
                        running = False
                        break

                    if early_stopping:
                        if early_stopping.early_stop(loss_val):
                            running = False
                            break
                iter_ += 1

        if config.training.save_best:
            save_model(model=model,
                       optimizer=loss_func,
                       epoch=epoch,
                       train_samples=-1,
                       file_name=os.path.join(file_name + str(i) + '.pt'),
                       save_path=os.path.join(config.model.save_path, writer_name),
                       model_state_dict=best_model_state,
                       optimizer_state_dict=best_optim_state)

    df = pd.DataFrame(pd_data)
    search = pd.DataFrame(search_res)

    if not os.path.exists(f'../Test/results/{writer_name}'):
        os.mkdir(f'../Test/results/{writer_name}')

    search.to_csv(f'../Test/results/{writer_name}/{file_name}_best.csv', index=False)
    df.to_csv(f'../Test/results/{writer_name}/{file_name}_all.csv', index=False)


if __name__ == '__main__':
    torch.manual_seed(666)

    parser = argparse.ArgumentParser(description='Training model for segmentation of PMSE signal')

    parser.add_argument('--config-path', type=str, default='models\\options\\train_generated_config.ymal',
                        help='Path to confg.ymal file (Default train_generated_config.ymal)')

    """
    unet_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                in_channels=3, out_channels=1, init_features=32, pretrained=True)

    from models.unets import UNet
    unet = UNet(3, 1, 32)

    unet.init_weights(unet_model.state_dict())
    torch.save({'model_state': unet.state_dict()}, 'weights\\mateuszbuda-brain-segmentation-pytorch.pt')
    """

    main(parser.parse_args())

