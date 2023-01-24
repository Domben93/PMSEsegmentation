import datetime
import os
import sys
import warnings

import torch
import torch.nn as nn
import torchsummary
from torch import Tensor
from collections import OrderedDict
from pathlib import Path
from itertools import chain
from typing import Tuple, Optional, Union, Literal, NoReturn, Callable, Any, Dict, List, TypeVar
from tqdm import tqdm
from base_model import BaseModel
import models.blocks as block

__all__ = ['UNet', 'ResUNet', 'save_model']

pad_modes = Literal['zeros', 'reflect', 'replicate', 'circular']
MODEL_EXTENSION = ['.pt', '.pth']


def save_model(model: nn.Module,
               optimizer: Any,
               epoch: int,
               train_samples: int,
               file_name: str,
               save_path: str = '',
               loss_history: Dict = None):
    state = {'sys_argv': sys.argv,
             'time': str(datetime.datetime.now()),
             'model_name': type(model).__name__,
             'model_state': model.state_dict(),
             'optimizer_name': type(optimizer).__name__,
             'optimizer_state': optimizer.state_dict(),
             'epoch': epoch,
             'totalTrainingSamples': train_samples,
             'loss': loss_history
             }

    if not Path(file_name).suffix in MODEL_EXTENSION:
        raise ValueError(f'File name must have one of the valid extensions: {MODEL_EXTENSION}. '
                         f'Got {Path(file_name).suffix}')

    if os.path.exists(save_path):
        full_path = os.path.join(save_path, file_name)
    else:
        warnings.warn(f'Could not find specified path: {save_path}. Saving at {os.getcwd()}')
        full_path = os.path.join(os.getcwd(), file_name)

    torch.save(state, full_path)


class UNet(nn.Module):

    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1,
                 initial_features: int = 32):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = initial_features

        # build encoder
        self.encoder1 = nn.Sequential(block.sequential_block_unet(in_channels, self.features))

        self.encoder2 = nn.Sequential(block.merge(block.pool2d(),
                                                  block.sequential_block_unet(self.features, self.features * 2)))

        self.encoder3 = nn.Sequential(block.merge(block.pool2d(),
                                                  block.sequential_block_unet(self.features * 2, self.features * 4)))

        self.encoder4 = nn.Sequential(block.merge(block.pool2d(),
                                                  block.sequential_block_unet(self.features * 4, self.features * 8)))
        # build bottleneck
        self.bottleneck = nn.Sequential(block.merge(block.pool2d(),
                                                    block.sequential_block_unet(self.features * 8, self.features * 16),
                                                    block.up_sample(self.features * 16, self.features * 8)))
        # build decoder
        self.decoder4 = nn.Sequential(block.merge(block.sequential_block_unet(self.features * 16, self.features * 8),
                                                  block.up_sample(self.features * 8, self.features * 4)))

        self.decoder3 = nn.Sequential(block.merge(block.sequential_block_unet(self.features * 8, self.features * 4),
                                                  block.up_sample(self.features * 4, self.features * 2)))

        self.decoder2 = nn.Sequential(block.merge(block.sequential_block_unet(self.features * 4, self.features * 2),
                                                  block.up_sample(self.features * 2, self.features)))

        self.decoder1 = nn.Sequential(block.sequential_block_unet(self.features * 2, self.features))
        # output layer
        self.final = nn.Sequential(block.merge(block.conv1x1(self.features, self.out_channels),
                                               block.final_layer(nn.Sigmoid)))

        self.layers = self._model_layers()

    def forward(self, x: Tensor) -> Tensor:

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(torch.cat((bottleneck, enc4), dim=1))
        dec3 = self.decoder3(torch.cat((dec4, enc3), dim=1))
        dec2 = self.decoder2(torch.cat((dec3, enc2), dim=1))
        dec1 = self.decoder1(torch.cat((dec2, enc1), dim=1))

        final = self.final(dec1)

        return final

    def init_weights(self, weights: Union[OrderedDict, List[Tensor], Callable], keys: List[str] = None) -> NoReturn:
        """
        Initiate model weights.

        Args:
            weights: OrderedDict, List of Tensors or List of Callable that initiates the weights.

            OrderedDict: Load state_dict of another model in its entirety. Requires that the models are of the same
            structure. Note that the keys of the ordered dict does not have to be the same as they are renamed
            to the keys of this model. Note also that weights and biases from e.g. BatchNormalization layers
            also will be given to the new model

            Tensors: If a list of tensors is passed then each nn.Conv2 layer will take those values in sequence. If keys
            are specified then the corresponding value pair to that key will be set and given the tensor weight.
            The number of weights must either be equal to the number of layers or the number of keys. Also, tensor
            weight dimension must be same the as models or errors will occur.

            Callable: If a  callables is passed then each nn.Conv2 layer will be initiated with that callable.
            In case of specified keys the method will only initiate the weights value pair to that key.

            keys: Models OrderedDict keys. Specify which layers to initiate

        """
        new_state_dict = self.state_dict()

        if isinstance(weights, OrderedDict):
            def convert_keys(loaded_state_dict: OrderedDict, new_state_dict: OrderedDict) -> OrderedDict:
                for load_key, new_key in zip(loaded_state_dict.keys(), new_state_dict.keys()):
                    new_state_dict[new_key] = loaded_state_dict[load_key]
                return new_state_dict

            new_state_dict = convert_keys(weights, new_state_dict)

        elif isinstance(weights, list):
            if not keys and not len(weights) == len(self.layers):
                raise ValueError(f'When keys are not specified then the number of weights in the list must be equal'
                                 f'to the number of layers.\nWas given {len(weights)} weights but have'
                                 f' {len(self.layers)} layers')

            if keys:
                assert len(keys) == len(weights)
                for key, weight in zip(keys, weights):
                    key += '.weight'
                    new_state_dict[key] = weight
            else:
                for i, layer_key in enumerate(self.layers):
                    layer_key += '.weight'
                    new_state_dict[layer_key] = weights[i]

        elif isinstance(weights, Callable):
            if keys:
                for key in keys:
                    key += '.weight'
                    new_state_dict[key] = weights(torch.empty(self.state_dict()[key].shape))
            else:
                for layer_key in self.layers:
                    layer_key += '.weight'
                    new_state_dict[layer_key] = weights(torch.empty(self.state_dict()[layer_key].shape))

        else:
            raise TypeError(f'Expected OrderedDict, List of Tensors or Callable. Got {type(weights)}')

        self.load_state_dict(new_state_dict)

    T = TypeVar('T')

    def freeze_layers(self, freeze_blocks: Union[str, List[str], List[int]] = None,
                      freeze_layers: List[T] = 'all') -> NoReturn:
        """
        Freeze layers
        Args:
            freeze_blocks:
            freeze_layers:
        """

        if freeze_blocks is None:
            return

        assert isinstance(freeze_blocks, (list, str)), f'"freeze_blocks must be a list, Got {type(freeze_blocks)}"'
        assert isinstance(freeze_layers, (list, str)), f'"freeze_blocks must be a list, Got {type(freeze_layers)}"'

        assert len(list(self.children())) >= len(
            freeze_blocks), f'Number of block elements to freeze is bigger than the' \
                            f'actual number of blocks in the model. Given {len(freeze_blocks)}' \
                            f' blocks to freeze but only have {len(list(self.children()))}' \
                            f' blocks in model'

        blocks = []

        if freeze_blocks == 'all':
            blocks = [x for x in range(len(list(self.children())))]

        elif all(isinstance(block, str) for block in freeze_blocks):
            for pos, block_name in enumerate(list(self.named_children())):
                for num, block_to_freeze in enumerate(freeze_blocks):
                    if block_name == block_to_freeze:
                        freeze_blocks.pop(num)
                        blocks.append(pos)
                        break

        elif all(isinstance(block, int) for block in freeze_blocks):
            blocks = freeze_blocks

        else:
            raise TypeError(f'Wrong composition of elements in list. Expected either list of str or int')

        for block_num in blocks:
            for pos, block_name in enumerate(list(self.named_children())):
                if block_num == pos:
                    if freeze_layers == 'all':
                        pass
                    break

        for num, child_module in enumerate(list(self.named_children())):

            if num not in blocks:
                continue

            named_layers = list(child_module[1].named_children())
            params = list(child_module[1].parameters())
            for layer, param in zip(named_layers, params):
                if isinstance(layer[1], (nn.Conv2d, nn.ConvTranspose2d)):
                    param.requires_grad = False
                    print(f'Freezing weigths: block "{child_module[0]} and layer "{layer[0]}" '
                          f'requires_grad set to {param.requires_grad}')

    def _model_layers(self):
        """
        Returns: layer names of the model
        """
        layer_names = []
        for blocks in list(self.named_children()):
            for layer in list(blocks[1].named_children()):
                if isinstance(layer[1], nn.Conv2d):
                    layer_names.append(str(blocks[0]) + '.' + str(layer[0]))

        return layer_names

    @property
    def number_of_layers(self):
        return len(self.layers)


class ResUNet(UNet):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 initial_features: int = 32):
        super(ResUNet, self).__init__(in_channels=in_channels,
                                      out_channels=out_channels,
                                      initial_features=initial_features)

    def forward(self, x: Tensor) -> Tensor:
        pass


"""

class ResUNet(nn.Module):

    # ResUNet architecture

    def __init__(self, channels: Tuple[int, int] = (1, 1),
                 initial_features: int = 32,
                 transfer_learning: bool = False
                 ) -> NoReturn:
        super(ResUNet, self).__init__()
        self.features = initial_features
        self.in_channels = channels[0]
        self.out_channels = channels[1]

        self.activation = nn.ReLU(inplace=True)

        # encoder modules with residual/skip connections
        self.encres1 = ResUNet._residual(in_channels=self.in_channels, out_channels=self.features)
        self.enc1 = ResUNet._sequential_block(in_channels=self.in_channels, out_channels=self.features,
                                              block_name='Encoder1')
        self.maxpool1 = pool2d()

        self.encres2 = ResUNet._residual(in_channels=self.features, out_channels=self.features * 2)
        self.enc2 = ResUNet._sequential_block(in_channels=self.features, out_channels=self.features * 2,
                                              block_name='Encoder2')
        self.maxpool2 = pool2d()

        self.encres3 = ResUNet._residual(in_channels=self.features * 2, out_channels=self.features * 4)
        self.enc3 = ResUNet._sequential_block(in_channels=self.features * 2, out_channels=self.features * 4,
                                              block_name='Encoder3')

        self.maxpool3 = pool2d()

        self.encres4 = ResUNet._residual(in_channels=self.features * 4, out_channels=self.features * 8)
        self.enc4 = ResUNet._sequential_block(in_channels=self.features * 4, out_channels=self.features * 8,
                                              block_name='Encoder4')
        self.maxpool4 = pool2d()
        # Bottleneck
        self.bottleneck_res = ResUNet._residual(in_channels=self.features * 8, out_channels=self.features * 16)
        self.bottleneck = ResUNet._sequential_block(in_channels=self.features * 8, out_channels=self.features * 16,
                                                    block_name='Bottleneck')

        # Decoder modules
        self.up_conv4 = up_sample(in_channels=self.features * 16, out_channels=self.features * 8)
        self.dec_res4 = ResUNet._residual(in_channels=self.features * 16, out_channels=self.features * 8)
        self.dec4 = ResUNet._sequential_block(in_channels=self.features * 16, out_channels=self.features * 8,
                                              block_name='Decoder4')

        self.up_conv3 = up_sample(in_channels=self.features * 8, out_channels=self.features * 4)
        self.dec_res3 = ResUNet._residual(in_channels=self.features * 8, out_channels=self.features * 4)
        self.dec3 = ResUNet._sequential_block(in_channels=self.features * 8, out_channels=self.features * 4,
                                              block_name='Decoder3')

        self.up_conv2 = up_sample(in_channels=self.features * 4, out_channels=self.features * 2)
        self.dec_res2 = ResUNet._residual(in_channels=self.features * 4, out_channels=self.features * 2)
        self.dec2 = ResUNet._sequential_block(in_channels=self.features * 4, out_channels=self.features * 2,
                                              block_name='Decoder2')

        self.up_conv1 = up_sample(in_channels=self.features * 2, out_channels=self.features)
        self.dec_res1 = ResUNet._residual(in_channels=self.features * 2, out_channels=self.features)
        self.dec1 = ResUNet._sequential_block(in_channels=self.features * 2, out_channels=self.features,
                                              block_name='Decoder1')

        self.output = ResUNet._residual(in_channels=self.features, out_channels=self.out_channels)

    def forward(self, x) -> Tensor:
        encoder1 = self.activation(self.enc1(x) + self.encres1(x))

        pool1 = self.maxpool1(encoder1)
        encoder2 = self.activation(self.enc2(pool1) + self.encres2(pool1))

        pool2 = self.maxpool2(encoder2)
        encoder3 = self.activation(self.enc3(pool2) + self.encres3(pool2))

        pool3 = self.maxpool3(encoder3)
        encoder4 = self.activation(self.enc4(pool3) + self.encres4(pool3))

        pool4 = self.maxpool4(encoder4)
        bottleneck = self.activation(self.bottleneck(pool4) + self.bottleneck_res(pool4))

        decoder4 = self.up_conv4(bottleneck)
        decoder4 = torch.concat((decoder4, encoder4), dim=1)
        decoder4 = self.activation(self.dec4(decoder4) + self.dec_res4(decoder4))

        decoder3 = self.up_conv3(decoder4)
        decoder3 = torch.concat((decoder3, encoder3), dim=1)
        decoder3 = self.activation(self.dec3(decoder3) + self.dec_res3(decoder3))

        decoder2 = self.up_conv2(decoder3)
        decoder2 = torch.concat((decoder2, encoder2), dim=1)
        decoder2 = self.activation(self.dec2(decoder2) + self.dec_res2(decoder2))

        decoder1 = self.up_conv1(decoder2)
        decoder1 = torch.concat((decoder1, encoder1), dim=1)
        decoder1 = self.activation(self.dec1(decoder1) + self.dec_res1(decoder1))

        output = self.output(decoder1)
        output = self.activation(output)

        return torch.sigmoid(output)

    @staticmethod
    def _sequential_block(in_channels: int, out_channels: int, block_name: str) -> nn.Sequential:
        # Using Sequential with OrderedDict.
        odict_ = OrderedDict([
            (
                block_name + '_Conv1', conv3x3(in_channels=in_channels,
                                               out_channels=out_channels,
                                               padding=1,
                                               padding_mode='replicate',
                                               bias=False)
            ),
            (block_name + '_batchNorm1', nn.BatchNorm2d(num_features=out_channels)),
            (block_name + '_ReLU1', nn.ReLU(inplace=True)),
            (
                block_name + '_Conv2', conv3x3(in_channels=out_channels,
                                               out_channels=out_channels,
                                               padding=1,
                                               padding_mode='replicate',
                                               bias=False)
            ),
            (block_name + '_BatchNorm2', nn.BatchNorm2d(num_features=out_channels)),
            (block_name + '_ReLU2', nn.ReLU(inplace=True))
        ])

        return nn.Sequential(odict_)

    @staticmethod
    def _residual(in_channels: int,
                  out_channels: int,
                  stride: int = 1,
                  padding: int = 0,
                  padding_mode: Optional[pad_modes] = 'replicate',
                  bias: bool = False
                  ) -> nn.Conv2d:
        return conv1x1(in_channels=in_channels,
                       out_channels=out_channels,
                       stride=stride,
                       padding=padding,
                       padding_mode=padding_mode,
                       bias=bias
                       )

"""