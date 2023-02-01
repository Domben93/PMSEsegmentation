import sys
import dotmap
import torch
import torch.nn as nn
import torch.optim as opt
from typing import Dict, Any
from utils import utils
from .unets import UNet, PSPUNet
import importlib

__all__ = ['change_weights_inputsize',
           'load_model',
           'load_optimizer']


def change_weights_inputsize(model: nn.Module, return_channel: int = 0) -> Dict:
    model_param = list(model.parameters())

    state_dict_copy = model.state_dict()

    if return_channel not in [0, 1, 2]:
        raise ValueError(f'Return channel must be 0, 1 or 2. Got {return_channel}.')

    if model_param[0].shape[-3] == return_channel + 1:
        return state_dict_copy

    first_block = list(model.named_children())[0]
    first_block_name = first_block[0]
    block_layers = first_block[1]

    if isinstance(block_layers, nn.Sequential):
        first_conv2layer_name = list(block_layers.state_dict())[0]
        for layer in block_layers:
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                key = first_block_name + '.' + first_conv2layer_name
                state_dict_copy[key].data = model.state_dict()[key].data[:, return_channel:return_channel + 1, :, :]
                print(state_dict_copy[key].data[:, :, :, :].shape)
                break

    return state_dict_copy


def load_model(config_path):
    config = utils.load_yaml_as_dotmap(config_path)

    if config.model.model_type == 'Unet':

        model = UNet(in_channels=config.model_init.in_channels,
                     out_channels=config.model_init.out_channels,
                     initial_features=config.model_init.init_features)

    elif config.model.model_type == 'PSPUnet':

        model = PSPUNet(in_channels=config.model_init.in_channels,
                        out_channels=config.model_init.out_channels,
                        initial_features=config.model_init.init_features,
                        pooling_size=(1, 2, 4, 6))
    else:
        raise NotImplementedError(f'{config.model.model_type} is not implemented')

    if config.model_init.pre_trained_weights:
        print(config.model_init.pre_trained_weights)
        old_model = torch.load(config.model_init.pre_trained_weights)

        if len(list(old_model['model_state'])) < len(list(model.state_dict())):

            if not list(old_model['model_state'])[:-2] == list(model.state_dict())[:len(list(old_model['model_state'])) - 2]:
                raise ValueError('Model state names must match between old and new model. To fix this load the'
                                 ' old model state dict into the model of same architecture and use init_weights.'
                                 ' This will change the state dict names such that one can use the weights.')

            model.init_weigths_by_layer(old_model['model_state'],
                                        [i for i in range(len(list(old_model['model_state'])) - 2)])

        elif len(list(old_model['model_state'])) == len(list(model.state_dict())):

            model.init_weights(old_model['model_state'])

        else:
            raise ValueError('')
    else:
        model.init_random_weights(model, init_layer=[nn.Conv2d, nn.ConvTranspose2d])

    if config.model_init.freeze_layers:
        model.freeze_blocks(config.model_init.freeze_layers, freeze_layer_types=[nn.Conv2d])

    return model


def load_optimizer(config: dotmap.DotMap, model: nn.Module, grad_true_only: bool = True) -> tuple[Any, Any]:
    if grad_true_only:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.parameters()

    if config.optimizer.optim_type == 'adam':

        optimizer = opt.Adam(params,
                             lr=config.optimizer.learning_rate,
                             weight_decay=config.optimizer.weight_decay,
                             betas=config.optimizer.betas)

    elif config.optimizer.optim_type == 'sgd':

        optimizer = opt.SGD(params,
                            lr=config.optimizer.learning_rate,
                            momentum=config.optimizer.momentum,
                            weight_decay=config.optimizer.weight_decay)

    else:
        raise ValueError(f'Optimizer {config.optimizer.optim_type} not implemented or does not exist')

    lr_scheduler = opt.lr_scheduler.StepLR(optimizer,
                                           step_size=config.learning_scheduler.step_size,
                                           gamma=config.learning_scheduler.gamma,
                                           last_epoch=config.learning_scheduler.last_epoch)
    return optimizer, lr_scheduler
