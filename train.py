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
        #[t.RandomHorizontalFlip(0.5)],
        #[t.RandomContrastAdjust(0.5, (0.8, 1.2))],
        #[t.RandomBrightnessAdjust(0.5, (0.5, 1.5))],
        #[t.ObjectAugmentation(augmentation_list)],
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

    # writer = SummaryWriter(log_dir=os.path.join(os.getcwd(), f'runs\\{type(model).__name__}', writer_name))
    # writer.add_graph(model, input_to_model=torch.rand(1, 3, 64, 64).to(device))
    # writer.add_text(tag='Run Info', text_string='Hey')

    iou_met = mIoU(reset_after_compute=True, device=device)

    best_model_state = {}
    best_optim_state = {}

    print(f'Training on device: {device}')

    pd_data = {
            'epoch': [],
            'run': [],
            'iou': [],
            'loss': [],
            'event': []
            }

    writer_name = f'{str(config.model.model_type)}_' \
                  f'pretrain-{str(config.model_init.pre_trained_weights)}_' \
                  f'freezed-{str(config.model_init.freeze_layers)}_' \
                  f'loss-{str(config.optimizer.loss_type)}_' \
                  f'optim-{str(config.optimizer.optim_type)}_' \
                  f'lr-{str(config.optimizer.learning_rate)}'

    if not os.path.exists(os.path.join(config.model.save_path, writer_name)):
        os.mkdir(os.path.join(config.model.save_path, writer_name))

    if config.training.number_of_runs < 1:
        raise ValueError(f'Number of runs must be bigger than 1. Got {config.training.number_of_runs}.')
    for i in range(config.training.number_of_runs):
        best_met = 0
        model = load_model(args.config_path)
        model.to(device)
        optimizer, lr_scheduler = load_optimizer(config, model, grad_true_only=True)

        loss_func = getattr(importlib.import_module('utils.loss'), config.optimizer.loss_type)()
        #loss_func = utils.loss.BinaryDiceLoss()
        #loss_func = torch.nn.BCELoss()

        print(f'Starting Training of {type(model).__name__} with \n'
              f'--> Optimizer: {type(optimizer).__name__}\n'
              f'--> loss function: {type(loss_func).__name__}\n'
              f'--> learning rate: {config.optimizer.learning_rate}\n',
              f'--> learning schedule: Step size: {config.learning_scheduler.step_size},'
              f' gamma: {config.learning_scheduler.gamma}, last epoch: {config.learning_scheduler.last_epoch} \n',
              f'--> Number of same model run: {i + 1} of {config.training.number_of_runs}')

        loss_train = []
        iou_train = []

        loss_val = []
        iou_val = []

        epoch = 0

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

                    iou_met(predicted, mask, info['image_name'])

                    train_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()

            loss_train.append(float(train_loss / len(train_loader)))
            iou_train.append(float(iou_met.compute()))

            pd_data['epoch'].append(epoch)
            pd_data['iou'].append(iou_train[-1])
            pd_data['loss'].append(loss_train[-1])
            pd_data['event'].append('Train')
            pd_data['run'].append(i)
            # iou_met.reset() # Called internally in iou_met.compute()

            # writer.add_scalar("Loss/train", loss_train[-1], epoch)
            # writer.add_scalar("mIoU/train", iou_train[-1], epoch)

            if validate_loader and (epoch % config.training.eval_interval == 0 or epoch == 1):
                model.eval()

                for data in validate_loader:

                    x, mask, info = data
                    x = x.to(device)
                    mask = mask.to(device)
                    with torch.no_grad():

                        predicted = model(x)

                        loss = loss_func(predicted, mask)
                        iou_met(predicted, mask, info['image_name'])

                        val_loss += loss.item()
                        """
                        if epoch % 5 == 0:
                            fig, ax = plt.subplots(1, 4)

                            iou = binary_jaccard_index(predicted, mask).item()
                            predicted = predicted.cpu().squeeze()

                            ax[0].imshow(x.cpu().squeeze()[0, :, :], cmap='jet', vmin=0, vmax=1)
                            ax[1].imshow(mask.cpu().squeeze(), cmap='jet', vmin=0, vmax=1)
                            ax[2].imshow(predicted, cmap='jet', vmin=0, vmax=1)
                            ax[3].imshow(torch.where(predicted >= 0.5, 1, 0), cmap='jet', vmax=1, vmin=0)
                            plt.title(f'IoU: {iou}')
                            plt.show()
                        """

                loss_val.append(float(val_loss / len(validate_loader)))
                iou_val.append(float(iou_met.compute()))

                pd_data['epoch'].append(epoch)
                pd_data['iou'].append(iou_val[-1])
                pd_data['loss'].append(loss_val[-1])
                pd_data['event'].append('Validation')
                pd_data['run'].append(i)

                # iou_met.reset() # Called internally in iou_met.compute()

                # writer.add_scalar('Loss/validate', loss_val[-1], epoch)
                # writer.add_scalar('mIoU/validate', iou_val[-1], epoch)

            if epoch == 1 or epoch % 1 == 0:
                print(f'{datetime.datetime.now()}, Epoch {epoch} of {config.training.epochs}:\n'
                      f'Training: loss {loss_train[-1]:.6f}, mIoU {iou_train[-1]:.6f}')
                if validate_loader:
                    print(f'Validation: loss {loss_val[-1]:.6f}, mIoU {iou_val[-1]:.6f}')

            if iou_val[-1] > best_met and config.training.save_best:
                best_met = iou_val[-1]
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
                       file_name=os.path.join('run_' + str(i) + '.pt'),
                       save_path=os.path.join(config.model.save_path, writer_name),
                       loss_history=hist_dict,
                       model_state_dict=best_model_state,
                       optimizer_state_dict=best_optim_state)

        # writer.close()
    df = pd.DataFrame(pd_data)
    df.to_csv(f'../Test/results/{writer_name}.csv', index=False)

    df = pd.DataFrame(pd_data)
    df.to_csv(f'../Test/results/{writer_name}.csv', index=False)
    sns.set_theme(style="darkgrid")

    sns.lineplot(x='epoch', y="iou",
                 hue="event",
                 data=df)

    plt.show()


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

