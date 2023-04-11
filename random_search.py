import importlib
import math
import os.path
import random
from typing import List

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
from utils.metrics import mIoU, SegMets
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.functional.classification import binary_jaccard_index


def main(args):
    print(f'Preparing training of model')

    config = load_yaml_as_dotmap(args.config_path)

    device = (torch.device(config.gpu) if torch.cuda.is_available() else torch.device('cpu'))

    augmentation_list = [t.RandomHorizontalFlip(0.25),
                         t.RandomVerticalFlip(0.25)]

    train_pair_compose = t.PairCompose([
        # [t.RandomHorizontalFlip(0.5)],
        # [t.RandomContrastAdjust(0.5, (0.8, 1.2))],
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

    pd_data = {}
    search_res = {}

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

    sample_classes = config.sample_classes.class_names

    writer_name = f'{str(config.model.model_type)}_{str(config.model_init.init_features)}_' \
                  f'pretrain-{bool(config.model_init.pre_trained_weights)}_' \
                  f'freezed-{str(config.model_init.freeze_layers)}_' \
                  f'loss-{str(config.optimizer.loss_type)}_' \
                  f'optim-{str(config.optimizer.optim_type)}'

    if not os.path.exists(os.path.join(config.model.save_path, writer_name)):
        os.mkdir(os.path.join(config.model.save_path, writer_name))

    if config.training.number_of_runs < 1:
        raise ValueError(f'Number of runs must be bigger than 1. Got {config.training.number_of_runs}.')

    lr_range = [0.01, 0.0001]
    weight_decay = [0.01, 0.0001]
    beta1 = [0.1, 0.0001]
    beta2 = config.optimizer.betas[1]

    param1: List = lr_range
    param2: List = weight_decay

    number_of_searches = 25
    validation_num = 1

    stop_patience = 10
    stop_min_delta = 0

    total_best_loss = math.inf

    for i in range(number_of_searches):

        lr = random.uniform(param1[0], param1[1])
        weight_decay = random.uniform(param2[0], param2[1])

        pd_data[i] = {}
        search_res[i] = {}

        variance_states = [{'model_state': None, 'optim_state': None} for i in range(validation_num)]
        found_best_flag = False

        for n in range(validation_num):
            best_loss_variance_run = math.inf

            search_res[i][n] = {'lr': lr,
                                'beta1': weight_decay,
                                'loss': 0,
                                'miou': 0,
                                'dsc': 0,
                                'auc': 0,
                                'acc': 0,
                                'epoch': 0
                                }
            pd_data[i][n] = {
                'epoch': [],
                'run': [],
                'iou': [],
                'dsc': [],
                'acc': [],
                'auc': [],
                'loss': [],
                'event': [],
                'lr': lr,
                'beta1': weight_decay,
            }

            model = load_model(args.config_path)
            model.to(device)
            optimizer, lr_scheduler = load_optimizer(config, model, grad_true_only=True, lr=lr,
                                                     weight_decay=weight_decay, betas=[0.9, 0.999])

            loss_func = getattr(importlib.import_module('utils.loss'), config.optimizer.loss_type)()

            print(f'Starting Training of {type(model).__name__} with \n'
                  f'--> Optimizer: {type(optimizer).__name__}\n'
                  f'--> Optimizer Params: {optimizer.defaults}'
                  f'--> loss function: {type(loss_func).__name__}\n'
                  f'--> learning schedule: Step size: {config.learning_scheduler.step_size},'
                  f' gamma: {config.learning_scheduler.gamma}, last epoch: {config.learning_scheduler.last_epoch} \n',
                  f'--> Number of same model run: {i + 1} of {number_of_searches}',
                  f'--> Number of same random values run: {n + 1} of {validation_num}')

            early_stopping = EarlyStopper(patience=stop_patience, min_delta=stop_min_delta)

            for epoch in range(1, config.training.epochs + 1):

                model.train()
                train_loss, val_loss = .0, .0

                for num, data in enumerate(train_loader):
                    x, mask, info = data
                    x = x.to(device)
                    mask = mask.to(device)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(mode=True):
                        predicted = model(x)

                        if isinstance(predicted, tuple):
                            predicted = predicted[0]

                        loss = loss_func(predicted, mask)

                        info = remove_from_dataname(info['image_name'])

                        metrics(predicted.detach().cpu(), mask.detach().cpu(), info)

                        train_loss += loss.item()

                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()

                loss_train = (float(train_loss / len(train_loader)))
                miou_train = float(metrics.mIoU(sample_classes=sample_classes))
                dsc_train = float(metrics.dice(sample_classes=sample_classes))
                acc_train = float(metrics.dice(sample_classes=sample_classes))
                auc_train = float(metrics.auc(sample_classes=sample_classes))
                metrics.reset()

                pd_data[i][n]['epoch'].append(epoch)
                pd_data[i][n]['iou'].append(miou_train)
                pd_data[i][n]['dsc'].append(dsc_train)
                pd_data[i][n]['acc'].append(acc_train)
                pd_data[i][n]['auc'].append(auc_train)
                pd_data[i][n]['loss'].append(loss_train)
                pd_data[i][n]['event'].append('Train')
                pd_data[i][n]['run'].append(n)

                if validate_loader and (epoch % config.training.eval_interval == 0 or epoch == 1):
                    model.eval()

                    for data in validate_loader:
                        x, mask, info = data
                        x = x.to(device)
                        mask = mask.to(device)
                        with torch.no_grad():
                            predicted = model(x)
                            if isinstance(predicted, tuple):
                                predicted = predicted[0]
                            loss = loss_func(predicted, mask)
                            info = remove_from_dataname(info['image_name'])

                            metrics(predicted.detach().cpu(), mask.detach().cpu(), info)

                            val_loss += loss.item()

                    loss_val = (float(val_loss / len(validate_loader)))
                    miou_val = float(metrics.mIoU(sample_classes=sample_classes))
                    dsc_val = float(metrics.dice(sample_classes=sample_classes))
                    acc_val = float(metrics.accuracy(sample_classes=sample_classes))
                    auc_val = float(metrics.auc(sample_classes=sample_classes))
                    metrics.reset()

                    pd_data[i][n]['epoch'].append(epoch)
                    pd_data[i][n]['iou'].append(miou_val)
                    pd_data[i][n]['loss'].append(loss_val)
                    pd_data[i][n]['dsc'].append(dsc_val)
                    pd_data[i][n]['acc'].append(acc_val)
                    pd_data[i][n]['auc'].append(auc_val)
                    pd_data[i][n]['event'].append('Validation')
                    pd_data[i][n]['run'].append(n)

                if epoch == 1 or epoch % 1 == 0:
                    print(f'{datetime.now()}, Epoch {epoch} of {config.training.epochs}:\n'
                          f'Training: loss {loss_train:.6f}, mIoU {miou_train:.6f}, dsc {dsc_train:.6f}, AUC {auc_train:.6f}')
                    if validate_loader:
                        print(
                            f'Validation: loss {loss_val:.6f}, mIoU {miou_val:.6f}, dsc {dsc_val:.6f}, AUC {auc_val:.6f}')

                if loss_val < best_loss_variance_run and config.training.save_best:
                    best_variance_run = loss_val
                    if total_best_loss > best_variance_run:
                        total_best_loss = best_variance_run
                        best_model_state = model.state_dict()
                        best_optim_state = optimizer.state_dict()
                        found_best_flag = True

                    variance_states[n]['model_state'] = model.state_dict()
                    variance_states[n]['optim_state'] = optimizer.state_dict()

                    search_res[i][n]['miou'] = miou_val
                    search_res[i][n]['dsc'] = dsc_val
                    search_res[i][n]['auc'] = auc_val
                    search_res[i][n]['acc'] = acc_val
                    search_res[i][n]['loss'] = loss_val
                    search_res[i][n]['epoch'] = epoch

                if early_stopping.early_stop(val_loss):
                    print('Stopping early!...')
                    break

        if found_best_flag:
            pass
            #for run, var_pair in enumerate(variance_states):

            #    save_model(model=model,
            #               optimizer=optimizer,
            #               epoch=-1,
            #               train_samples=-1,
            #               file_name=os.path.join(f'variance_run_{run}.pt'),
            #               save_path=os.path.join(config.model.save_path, writer_name),
            #               model_state_dict=var_pair['model_state'],
            #               optimizer_state_dict=var_pair['optim_state']
            #               )

    if config.training.save_best:
        save_model(model=model,
                   optimizer=optimizer,
                   epoch=-1,
                   train_samples=-1,
                   file_name=os.path.join('best_random_search.pt'),
                   save_path=os.path.join(config.model.save_path, writer_name),
                   model_state_dict=best_model_state,
                   optimizer_state_dict=best_optim_state)

    df = pd.DataFrame(pd_data)
    search = pd.DataFrame(search_res)

    search.to_csv(f'../Test/results/{writer_name + "_random_search"}.csv', index=False)
    df.to_csv(f'../Test/results/{writer_name}.csv', index=False)


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
