import torch
import torch.nn as nn
from typing import Dict
from collections import OrderedDict


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
            if isinstance(layer, nn.Conv2d):
                key = first_block_name + '.' + first_conv2layer_name
                state_dict_copy[key].data = model.state_dict()[key].data[:, return_channel:return_channel + 1, :, :]
                print(state_dict_copy[key].data[:, :, :, :].shape)
                break

    return state_dict_copy

