import importlib
import math
import os.path
import torch
import argparse
import datetime
import pandas as pd
import utils.loss
from datetime import datetime
from utils import transforms as t
from utils.dataset import get_dataloader
from utils.utils import *
from models.utils import *
from utils.metrics import SegMets
from copy import deepcopy


def main(args):
    print(f'Preparing training of model')

    config = load_yaml_as_dotmap(args.config_path)

    device = (torch.device(config.gpu) if torch.cuda.is_available() else torch.device('cpu'))

    train_pair_compose = t.PairCompose([
        # [t.RandomVerticalFlip(0.5)],
        # [t.RandomHorizontalFlip(0.5)],
        # [t.RandomContrastAdjust(0.5, (0.8, 1.2))],
        [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)],
        [t.Normalize((0, 1), (0, 255), return_type=torch.float32), None],
        [t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale, padding_mode='constant'),
        t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale, padding_mode='constant')]
    ])

    val_pair_compose = t.PairCompose([
        [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)],
        [t.Normalize((0, 1), (0, 255), return_type=torch.float32), None],
        [t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale, padding_mode='constant'),
         t.QuasiResize(config.dataset.resize_shape, config.dataset.max_scale, padding_mode='constant')]
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

    ds = '_Deep-supervision' if config.model_init.deep_supervision else ''
    aug = '_aug-Hflip-Cadj_'

    # Folder name
    writer_name = f'{str(config.model.model_type)}_{str(config.model_init.init_features)}_' \
                  f'pretrain-{bool(config.model_init.pre_trained_weights)}_' \
                  f'loss-{str(config.optimizer.loss_type)}_' \
                  f'optim-{str(config.optimizer.optim_type)}' \
                  f'' + ds + aug
    # File name
    file_name = f'lr_{config.optimizer.learning_rate}_' \
                f'wd_{config.optimizer.weight_decay}_' \
                f'betas_{config.optimizer.betas[0]}-{config.optimizer.betas[1]}_' \
                f'momentum_{config.optimizer.momentum}_' \
                f'freezed-{"".join([str(x) for x in config.model_init.freeze_layers]) if config.model_init.freeze_layers is not None else None}_'\

    if not os.path.exists(os.path.join(config.model.save_path, writer_name)):
        os.mkdir(os.path.join(config.model.save_path, writer_name))

    if config.training.number_of_runs < 1:
        raise ValueError(f'Number of runs must be bigger than 1. Got {config.training.number_of_runs}.')

    for i in range(config.training.number_of_runs):

        best_met = math.inf
        model = load_model(args.config_path)

        model.to(device)
        optimizer, lr_scheduler = load_optimizer(config, model, grad_true_only=True,
                                                 lr=config.optimizer.learning_rate,
                                                 weight_decay=config.optimizer.weight_decay,
                                                 betas=config.optimizer.betas,
                                                 momentum=config.optimizer.momentum)

        loss_func = getattr(importlib.import_module('utils.loss'), config.optimizer.loss_type)()

        #loss_func = utils.loss.DiceSurfaceLoss(strategy='rebalance', inc_reb_rate_iter=5)

        print(f'Starting Training of {type(model).__name__} with \n'
              f'--> Optimizer: {type(optimizer).__name__}\n'
              f'--> Optimizer Params: {optimizer.defaults}'
              f'--> loss function: {type(loss_func).__name__}\n'
              f'--> learning schedule: Step size: {config.learning_scheduler.step_size},'
              f' gamma: {config.learning_scheduler.gamma}, last epoch: {config.learning_scheduler.last_epoch} \n',
              f'--> Number of same model run: {i + 1} of {config.training.number_of_runs}')


        iter = 0
        epoch = 0
        if config.training.early_stopping.early_stop:
            early_stopping = EarlyStopper(patience=config.training.early_stopping.patience,
                                          min_delta=config.training.early_stopping.min_delta)
        else:
            early_stopping = False

        for epoch in range(1, config.training.epochs + 1):

            model.train()
            train_loss, val_loss = .0, .0

            for num, data in enumerate(train_loader):
                iter += 1
                x, mask, info = data
                x = x.to(device)
                mask = mask.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(mode=True):

                    predicted = model(x)

                    if isinstance(predicted, tuple):
                        if len(predicted) == 1:
                            predicted = predicted[0]
                        elif config.model_init.avg_output:
                            predicted = (predicted[0] + predicted[1] + predicted[2] + predicted[3]) / len(predicted)
                        else:
                            predicted = predicted
                    info = remove_from_dataname_extended(info['image_name'], sample_classes)

                    if isinstance(predicted, tuple) and config.model_init.deep_supervision:
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

                        predicted = (predicted[0] + predicted[1] + predicted[2] + predicted[3]) / len(predicted)
                        metrics(predicted.detach().cpu(), mask.detach().cpu(), info)
                        train_loss += ((losses[0].item() +
                                        losses[1].item() +
                                        losses[2].item() +
                                        losses[3].item()) / len(losses))
                    else:
                        loss = loss_func(predicted, mask)

                        metrics(predicted.detach().cpu(), mask.detach().cpu(), info)

                        train_loss += loss.item()

                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()

            loss_train = (float(train_loss / len(train_loader)))
            miou_train = float(metrics.mIoU(sample_classes=sample_classes, multidim_average='global'))
            dsc_train = float(metrics.dice(sample_classes=sample_classes, multidim_average='global'))

            metrics.reset()

            pd_data['epoch'].append(epoch)
            pd_data['iou'].append(miou_train)
            pd_data['dsc'].append(dsc_train)
            pd_data['loss'].append(loss_train)
            pd_data['event'].append('Train')
            pd_data['run'].append(i)

            if validate_loader and (epoch % config.training.eval_interval == 0 or epoch == 1):
                model.eval()

                for data in validate_loader:

                    x, mask, info = data
                    x = x.to(device)
                    mask = mask.to(device)
                    with torch.no_grad():

                        predicted = model(x)

                        if config.model_init.deep_supervision:
                            if config.model_init.avg_output:
                                predicted = (predicted[0] + predicted[1] + predicted[2] + predicted[3]) / len(predicted)
                            else:
                                predicted = predicted[0]

                        loss = loss_func(predicted, mask)
                        info = remove_from_dataname_extended(info['image_name'], sample_classes)

                        metrics(predicted.detach().cpu(), mask.detach().cpu(), info)

                        val_loss += loss.item()

                loss_val = (float(val_loss / len(validate_loader)))
                miou_val = float(metrics.mIoU(sample_classes=sample_classes, multidim_average='global'))
                dsc_val = float(metrics.dice(sample_classes=sample_classes, multidim_average='global'))
                metrics.reset()

                pd_data['epoch'].append(epoch)
                pd_data['iou'].append(miou_val)
                pd_data['loss'].append(loss_val)
                pd_data['dsc'].append(dsc_val)
                pd_data['event'].append('Validation')
                pd_data['run'].append(i)

            if epoch == 1 or epoch % 1 == 0:
                print(f'{datetime.now()}, Epoch {epoch} of {config.training.epochs}, iter {iter}:\n'
                      f'Training: loss {loss_train:.6f}, mIoU {miou_train:.6f}, dsc {dsc_train:.6f}')
                if validate_loader:
                    print(f'Validation: loss {loss_val:.6f}, mIoU {miou_val:.6f}, dsc {dsc_val:.6f}')

            if loss_val < best_met and config.training.save_best:
                best_met = loss_val
                best_model_state = deepcopy(model.state_dict())
                best_optim_state = deepcopy(optimizer.state_dict())

            if early_stopping:
                if early_stopping.early_stop(loss_val):
                    break

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

    if not os.path.exists(f'../PMSE-segmentation/results/{writer_name}'):
        os.mkdir(f'../PMSE-segmentation/results/{writer_name}')

    df.to_csv(f'../PMSE-segmentation/results/{writer_name}/{file_name}_all.csv', index=False)


if __name__ == '__main__':
    torch.manual_seed(666)

    parser = argparse.ArgumentParser(description='Training model for segmentation of PMSE signal')

    parser.add_argument('--config-path', type=str, default='config\\unet_config.ymal',
                        help='Path to confg.ymal file (Default unet_config.ymal)')
    main(parser.parse_args())
