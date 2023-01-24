import torch
import torch.nn as nn
import torchsummary
from torch import Tensor
from typing import Tuple, Literal, Callable
from models.base_model import BaseModel
import models.blocks as block
import yaml

__all__ = ['UNet', 'ResUNet', 'PSPUNet']

pad_modes = Literal['zeros', 'reflect', 'replicate', 'circular']


class UNet(BaseModel):

    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1,
                 initial_features: int = 32,
                 activation_func: Callable = nn.Sigmoid):
        super(UNet, self).__init__(in_channels=in_channels,
                                   out_channels=out_channels)

        self.features = initial_features

        # build encoder
        self.encoder1 = nn.Sequential(block.sequential_block_unet(self.in_channels, self.features))

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
                                               block.final_layer(activation_func)))

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


class PSPUNet(BaseModel):

    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1,
                 initial_features: int = 32,
                 pooling_size: Tuple[int, ...] = (1, 2, 4, 8)):
        super(PSPUNet, self).__init__(in_channels=in_channels,
                                      out_channels=out_channels)

        self.features = initial_features

        # build encoder
        self.encoder1 = nn.Sequential(block.sequential_block_unet(self.in_channels, self.features))

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

        self.psppooling = block.PSPPooling(self.features, self.features, poolingsizes=pooling_size)

        self.final = nn.Sequential(block.merge(block.conv1x1(self.features, self.out_channels),
                                               block.final_layer(nn.Sigmoid)))

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

        psppool = self.psppooling(dec1)

        final = self.final(psppool)

        return final


class ResUNet(BaseModel):

    def __init__(self):
        super(ResUNet, self).__init__()

        self.encoder1 = nn.Sequential()

    def forward(self, x: Tensor) -> Tensor:
        pass


class ResUNetA(nn.Module):
    pass


if __name__ == '__main__':
    from models.utils import change_weights_inputsize
    torch.manual_seed(666)

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=True)

    unet = UNet(1, 1, 32)
    print(unet.encoder1.Conv1.weight.shape)

    # params = [p for p in model.parameters() if p.requires_grad]

    #torchsummary.summary(model.cuda(), (3, 64, 64))

    print(model.encoder1.enc1conv1.weight.shape)

    state_dict = change_weights_inputsize(model, 0)

    unet.init_weights(state_dict)
    unet.freeze_blocks([0, 1, 2, 3, 5, 6, 7, 8])
    torchsummary.summary(unet.cuda(), (1, 64, 64))
    """
    model_size = len(list(model.state_dict()))

    pspunet = PSPUNet(3, 1, 32).cuda()

    print(pspunet.get_parameter('encoder1.Conv1.weight'))

    pspunet.init_pretrained_weights(model, [i for i in range(model_size - 2)])

    print(pspunet.get_parameter('encoder1.Conv1.weight'))

    #pspunet.freeze_blocks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], freeze_layer_types=[nn.Conv2d, nn.ConvTranspose2d])
    #freeze_layers(freeze_blocks='all')
    # print(pspunet.get_submodule('encoder1'))
    """
