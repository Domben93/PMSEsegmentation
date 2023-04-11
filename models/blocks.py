""" Unet building blocks"""
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor
from collections import OrderedDict
from itertools import chain
from typing import Literal, Optional, Union, Tuple, Callable

pad_modes = Literal['zeros', 'reflect', 'replicate', 'circular']


def conv3x3(in_channels: int,
            out_channels: int,
            stride: Optional[Union[int, Tuple[int, ...]]] = 1,
            padding: Optional[Union[int, Tuple[int, ...]]] = 0,
            padding_mode: Optional[pad_modes] = 'zeros',
            dilation: Optional[Union[int, Tuple[int, ...]]] = 1,
            groups: Optional[int] = 1,
            bias: Optional[bool] = True
            ) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(3, 3),
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        groups=groups,
        bias=bias
    )


def conv1x1(in_channels: int,
            out_channels: int,
            stride: Optional[Union[int, Tuple[int, ...]]] = 1,
            padding: Optional[Union[int, Tuple[int, ...]]] = 0,
            padding_mode: Optional[pad_modes] = 'zeros',
            dilation: Optional[Union[int, Tuple[int, ...]]] = 1,
            groups: Optional[int] = 1,
            bias: Optional[bool] = True
            ) -> OrderedDict:
    return OrderedDict([('Conv', nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(1, 1),
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        dilation=dilation,
        groups=groups,
        bias=bias
    ))])


pooling_modes = Literal['max', 'average']


def pool2d(kernel_size: int = 2,
           stride: int = 2,
           pooling_mode: Optional[pooling_modes] = 'max'
           ) -> OrderedDict:
    pool_types = {
        'max': nn.MaxPool2d(kernel_size=kernel_size, stride=stride),
        'average': nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    }
    return OrderedDict([(f'{pooling_mode}pool', pool_types[pooling_mode])])


def up_sample(in_channels: int,
              out_channels: int,
              kernel_size: int = 2,
              stride: int = 2) -> OrderedDict:
    return OrderedDict([('ConvTrans', nn.ConvTranspose2d(in_channels=in_channels,
                                                         out_channels=out_channels,
                                                         kernel_size=(kernel_size, kernel_size),
                                                         stride=(stride, stride)
                                                         ))])

def sequential_block_unet(in_channels, out_channels, drop_out=False, p=0.5) -> OrderedDict:
    """

    Args:
        p:
        drop_out:
        in_channels:
        out_channels:
        block_name:

    Returns:

    """

    odict_ = OrderedDict([
        (
            'Conv1', conv3x3(in_channels=in_channels,
                             out_channels=out_channels,
                             padding=1,
                             padding_mode='zeros',
                             bias=True)
        ),
        ('BatchNorm1', nn.BatchNorm2d(num_features=out_channels)),
        ('ReLU1', nn.ReLU(inplace=True)),
        (
            'Conv2', conv3x3(in_channels=out_channels,
                             out_channels=out_channels,
                             padding=1,
                             padding_mode='zeros',
                             bias=True)
        ),
        ('BatchNorm2', nn.BatchNorm2d(num_features=out_channels)),
        ('ReLU2', nn.ReLU(inplace=True))
    ])

    if drop_out:
        odict_.update([('DropOut', nn.Dropout2d(p=p))])

    return odict_


def sequentail_block_unetplusspluss(in_channels, out_channels, drop_out=False, p=0.5):
    odict_ = OrderedDict([
        (
            'Conv1', conv3x3(in_channels=in_channels,
                             out_channels=out_channels,
                             padding=1,
                             padding_mode='zeros',
                             bias=True)
        ),
        ('BatchNorm1', nn.BatchNorm2d(num_features=out_channels)),
        ('ReLU1', nn.LeakyReLU(inplace=True)),
        (
            'Conv2', conv3x3(in_channels=out_channels,
                             out_channels=out_channels,
                             padding=1,
                             padding_mode='zeros',
                             bias=True)
        ),
        ('BatchNorm2', nn.BatchNorm2d(num_features=out_channels)),
        ('ReLU2', nn.LeakyReLU(inplace=True))
    ])

    if drop_out:
        odict_.update([('DropOut', nn.Dropout2d(p=p))])

    return odict_


def sequential_block_resunet(in_channels, out_channels, drop_out=False, p=0.5) -> OrderedDict:
    odict_ = OrderedDict([

        ('BatchNorm1', nn.BatchNorm2d(num_features=out_channels)),
        ('ReLU1', nn.ReLU(inplace=True)),
        ('Conv1', conv3x3(in_channels=in_channels,
                          out_channels=out_channels,
                          padding=1,
                          padding_mode='replicate',
                          bias=False)),

        ('BatchNorm2', nn.BatchNorm2d(num_features=out_channels)),
        ('ReLU2', nn.ReLU(inplace=True)),
        ('Conv2', conv3x3(in_channels=out_channels,
                          out_channels=out_channels,
                          padding=1,
                          padding_mode='replicate',
                          bias=False))
    ])

    if drop_out:
        odict_.update([('DropOut', nn.Dropout2d(p=p))])

    return odict_


def final_layer(activation: Callable) -> OrderedDict:
    return OrderedDict([(f'{activation.__name__}',
                         activation())])


def merge(*odicts: OrderedDict) -> OrderedDict:
    return OrderedDict(chain.from_iterable([x.items() for x in odicts]))


from typing import TypeVar

T = TypeVar('T', bound='Module')


class Upsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(Upsample, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return f.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode)


class ResidualConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualConv, self).__init__()

        self.main_branch = nn.Sequential(sequential_block_resunet(in_channels=in_channels,
                                                                  out_channels=out_channels,
                                                                  ))
        self.skip_branch = nn.Sequential()

    def forward(self, x: Tensor) -> Tensor:
        pass


# Based on Lextal's code found at https://github.com/Lextal/pspnet-pytorch/blob/master/pspnet.py

class PSPPooling(nn.Module):

    def __init__(self, in_features, out_features, poolingsizes: Tuple[int, ...] = (1, 2, 4, 8)):
        super(PSPPooling, self).__init__()

        self.pooling_pyramid = nn.ModuleList()
        for size in poolingsizes:
            self.pooling_pyramid.append(self._make_pyramid_level(in_features, size))

        self.pyramid_output = nn.Sequential(OrderedDict([
            (
            f'outConv', nn.Conv2d(in_features * (len(poolingsizes) + 1), out_features, kernel_size=(1, 1), bias=False)),
            (f'outBN', nn.BatchNorm2d(num_features=out_features)),
            (f'outReLU', nn.ReLU())
        ]))

    def _make_pyramid_level(self, features, size):
        pyramid_layer = OrderedDict([
            (f'AvgPool{size}x{size}', nn.AdaptiveAvgPool2d(output_size=(size, size))),
            (f'Conv1', nn.Conv2d(features, features, kernel_size=(1, 1), bias=False))
        ])
        return nn.Sequential(pyramid_layer)

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]

        priors = [f.upsample_bilinear(input=layer(x), size=(h, w)) for layer in self.pooling_pyramid] + [x]
        out = self.pyramid_output(torch.cat(priors, dim=1))
        return out


class UNetEncoderBlock(nn.Module):
    """
    UNet Encoder block
    Encoder block for a UNet model with two convolutional layers where we apply batch normalization and ReLU activation
    after each. NOTE! The max-pooling operation is done before the sequential UNet block. This is done in order to
    pass the output of the block as a skip connection as well as a input to the next block.
    For the first block the "first" parameter can be set to True in order to skip the max-pooling operation

                                                                                In:                  Out:

    Input:                         Tensor                                           (In_channels, H, W)
                                     |
                                     ∨
    Pool:                         MaxPool                               (Out_channels, H, W)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    Conv1:                        Conv 3x3                            (in_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    BN1:                   Batch Normalization                       (Out_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    ReLU1:                    ReLU {inplace}                         (Out_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    Conv2:                        Conv 3x3                           (Out_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    BN2:                   Batch Normalization                       (Out_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    ReLU2:                     ReLU {inplace}                        (Out_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    Output:                        Tensor                                       (Out_channels, H/2, W/2)


    """

    def __init__(self, in_channels, out_channels, block_name, first=False):
        super(UNetEncoderBlock, self).__init__()

        self.block_dict = sequential_block_unet(in_channels, out_channels, block_name)
        if not first:
            self.block_dict.update({block_name + 'MaxPool': pool2d()})
            self.block_dict.move_to_end(block_name + 'MaxPool', last=False)
        self.block = nn.Sequential(self.block_dict)

    def forward(self, x) -> torch.Tensor:
        return self.block(x)


class UNetDecoderBlock(nn.Module):
    """
    UNet Decoder block
    Decoder block for a UNet model with two convolutional layers where we apply batch normalization and ReLU activation
    after each. at the end of the block we up-sample using transposed convolution

                                                                                In:                 Out:

    Input:                         Tensor                              (In_channels, H, W)
                                     |
                                     ∨
    Pool:                         MaxPool                             (in_channels, H, W)   (in_channels, H/2, W/2)
                                     |
                                     ∨
    Conv1:                        Conv 3x3                           (in_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    BN1:                   Batch Normalization                       (Out_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    ReLU1:                    ReLU {inplace}                         (Out_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    Conv2:                        Conv 3x3                           (Out_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    BN2:                   Batch Normalization                       (Out_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    ReLU2:                     ReLU {inplace}                        (Out_channels, H/2, W/2)   (Out_channels, H/2, W/2)
                                     |
                                     ∨
    Up sample:                  ConvTrans 2x2                          (out_channels, H, W)     (in_channels, H*2, W*2)
                                     |
                                     ∨
    Output:                        Tensor                                                       (Out_channels, H*2, W*2)



    """

    def __init__(self, in_channels, out_channels, block_name, last=False):
        super(UNetDecoderBlock, self).__init__()

        self.block_dict = sequential_block_unet(in_channels * 2, in_channels, block_name)
        if not last:
            self.block_dict.update({block_name + 'ConvTranspose': up_sample(in_channels, out_channels)})

        self.block = nn.Sequential(self.block_dict)

    def forward(self, x, skip_connection) -> torch.Tensor:
        x = torch.cat((x, skip_connection), dim=1)
        return self.block(x)


class UNetBottleneckBlock(nn.Module):
    """
    UNet bottleneck block

                                                                                    In:                 Out:

    Input:                         Tensor                              (In_channels, H, W)
                                     |
                                     ∨
    Conv1:                        Conv 3x3                           (Out_channels, H, W)   (Out_channels, H, W)
                                     |
                                     ∨
    BN1:                   Batch Normalization                       (Out_channels, H, W)   (Out_channels, H, W)
                                     |
                                     ∨
    ReLU1:                    ReLU {inplace}                         (Out_channels, H, W)   (Out_channels, H, W)
                                     |
                                     ∨
    Conv2:                        Conv 3x3                           (Out_channels, H, W)   (Out_channels, H, W)
                                     |
                                     ∨
    BN2:                   Batch Normalization                       (Out_channels, H, W)   (Out_channels, H, W)
                                     |
                                     ∨
    ReLU2:                     ReLU {inplace}                        (Out_channels, H, W)   (Out_channels, H, W)
                                     |
                                     ∨
    Up sample:                  ConvTrans 2x2                          (In_channels, H, W)     (Out_channels, H*2, W*2)
                                     |
                                     ∨
    Output:                        Tensor                                                      (Out_channels, H*2, W*2)

    """

    def __init__(self, in_channels, out_channels, block_name):
        super(UNetBottleneckBlock, self).__init__()
        self.block_dict = sequential_block_unet(in_channels, out_channels, block_name)
        self.block_dict.update({block_name + '-MaxPool': pool2d()})
        self.block_dict.update({block_name + 'ConvTranspose': up_sample(out_channels, in_channels)})
        self.block_dict.move_to_end(block_name + '-MaxPool', last=False)

        self.block = nn.Sequential(self.block_dict)

    def forward(self, x):
        return self.block(x)


class FinalBlock(nn.Module):
    """
    Output block layer.

                                                                                       In:                 Out:

    Input:                         Tensor                              (In_channels, H, W)
                                     |
                                     ∨
    Conv1:                        Conv 1x1                             (in_channels, H, W)   (Out_channels, H, W)
                                     |
                                     ∨
    activation:             Activation {Callable}                      (Out_channels, H, W)   (Out_channels, H, W)
                                     |
                                     ∨
    Output:                        Tensor                                                     (Out_channels, H, W)
    """

    def __init__(self, in_channels, out_channels, block_name, activation: Callable):
        super(FinalBlock, self).__init__()
        self.activation = activation
        self.final = nn.Sequential(OrderedDict([(block_name + '_Conv1x1', conv1x1(in_channels, out_channels)),
                                                (block_name + f'_{self.activation.__name__}',
                                                 self.activation())]))

    def forward(self, x):
        return self.final(x)
