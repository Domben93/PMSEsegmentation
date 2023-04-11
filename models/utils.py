import sys
import dotmap
import torch
import torch.nn as nn
import torch.optim as opt
from typing import Dict, Any, Union, List, Tuple
from utils import utils
from .unets import UNet, PSPUNet, UNet_vgg, UNet_unet
from .unet_plusspluss.unet_plusspluss import Generic_UNetPlusPlus
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

    elif config.model.model_type == 'Unet_vgg':
        model = UNet_vgg(in_channels=config.model_init.in_channels,
                         out_channels=config.model_init.out_channels,
                         initial_features=config.model_init.init_features)

    elif config.model.model_type == 'Unet_unet':
        model = UNet_unet(in_channels=config.model_init.in_channels,
                          out_channels=config.model_init.out_channels,
                          initial_features=config.model_init.init_features)

    elif config.model.model_type == 'PSPUnet':

        model = PSPUNet(in_channels=config.model_init.in_channels,
                        out_channels=config.model_init.out_channels,
                        initial_features=config.model_init.init_features,
                        pooling_size=(1, 2, 4, 8))

    elif config.model.model_type == 'Unet_plusspluss':
        model = Generic_UNetPlusPlus(input_channels=config.model_init.in_channels,
                                     base_num_features=config.model_init.init_features,
                                     num_classes=1,
                                     num_pool=4,
                                     convolutional_pooling=False,
                                     convolutional_upsampling=True,
                                     deep_supervision=config.model_init.deep_supervision,
                                     init_encoder=config.model_init.pre_init_weights,
                                     seg_output_use_bias=True)
    else:
        raise NotImplementedError(f'{config.model.model_type} is not implemented')

    if config.model_init.pre_trained_weights and not config.model_init.fine_tune:

        old_model = torch.load(config.model_init.pre_trained_weights)

        if len(list(old_model['model_state'])) < len(list(model.state_dict())):

            if not list(old_model['model_state'])[:-2] == list(model.state_dict())[
                                                          :len(list(old_model['model_state'])) - 2]:
                raise ValueError('Model state names must match between old and new model. To fix this load the'
                                 ' old model state dict into the model of same architecture and use init_weights.'
                                 ' This will change the state dict names such that one can use the weights.')

            model.init_weigths_by_layer(old_model['model_state'],
                                        [i for i in range(len(list(old_model['model_state'])) - 2)])

        elif len(list(old_model['model_state'])) == len(list(model.state_dict())):

            model.init_weights(old_model['model_state'])

        else:
            raise ValueError('')

    elif config.model_init.fine_tune:
        model.load_state_dict(torch.load(config.model_init.pre_trained_weights)['model_state'])

    elif config.model_init.pre_initiated:
        pass

    else:
        model.init_random_weights(model, init_layer=[nn.Conv2d, nn.ConvTranspose2d])

    if config.model_init.freeze_layers:
        model.freeze_blocks(config.model_init.freeze_layers, freeze_layer_types=[nn.Conv2d])

    return model


def load_optimizer(config: dotmap.DotMap,
                   model: nn.Module,
                   grad_true_only: bool = True,
                   lr: float = None,
                   weight_decay: float = None,
                   betas: Union[Tuple, List] = None,
                   momentum: float = None) -> Tuple[Any, Any]:
    if not lr:
        lr = config.optimizer.learning_rate
    if not weight_decay:
        weight_decay = config.optimizer.weight_decay
    if not betas:
        betas = config.optimizer.betas
    if not momentum:
        momentum = config.optimizer.momentum

    if grad_true_only:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.parameters()

    if config.optimizer.layer_learning_rate:
        pass

    if config.optimizer.optim_type == 'adam':

        optimizer = opt.Adam(params,
                             lr=lr,
                             weight_decay=weight_decay,
                             betas=betas)

    elif config.optimizer.optim_type == 'adamW':

        optimizer = opt.AdamW(params,
                              lr=lr,
                              weight_decay=weight_decay,
                              betas=betas)

    elif config.optimizer.optim_type == 'adam_amsgrad':
        optimizer = opt.Adam(params,
                             lr=lr,
                             weight_decay=weight_decay,
                             betas=betas,
                             amsgrad=True)

    elif config.optimizer.optim_type == 'adamW_amsgrad':

        optimizer = opt.AdamW(params,
                              lr=lr,
                              weight_decay=weight_decay,
                              betas=betas,
                              amsgrad=True)

    elif config.optimizer.optim_type == 'sgd':

        optimizer = opt.SGD(params,
                            lr=lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

    else:
        raise ValueError(f'Optimizer {config.optimizer.optim_type} not implemented or does not exist')

    lr_scheduler = opt.lr_scheduler.StepLR(optimizer,
                                           step_size=config.learning_scheduler.step_size,
                                           gamma=config.learning_scheduler.gamma,
                                           last_epoch=config.learning_scheduler.last_epoch)
    return optimizer, lr_scheduler
