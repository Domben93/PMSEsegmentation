import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchsummary
from torch import Tensor
from typing import Tuple, Literal, Callable, List
from models.base_model import BaseModel
import models.blocks as block
import torchvision.models as model
import torch
import yaml

__all__ = ['UNet', 'ResUNet', 'PSPUNet', 'UNet_vgg', 'UNet_unet']

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


class UNet_unet(UNet):

    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1,
                 initial_features: int = 32,
                 activation_func: Callable = nn.Sigmoid):
        super(UNet_unet, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        initial_features=initial_features,
                                        activation_func=activation_func)

        self.init_from_pretrained()

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

    def init_from_pretrained(self):

        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=3, out_channels=1, init_features=32, pretrained=True)

        seq = model.children()

        seq_list_old = []
        for i in seq:
            if isinstance(i, nn.Sequential):
                for child in i.children():
                    seq_list_old.append(child)
            elif isinstance(i, (nn.ConvTranspose2d, nn.MaxPool2d)):
                seq_list_old.append(i)

        seq_list_new = []
        for i in self.children():
            if isinstance(i, nn.Sequential):
                for child in i.children():
                    seq_list_new.append(child)
            elif isinstance(i, (nn.ConvTranspose2d, nn.MaxPool2d)):
                seq_list_new.append(i)

        seq_list_new = seq_list_new[:-2]

        for new, old in zip(seq_list_new, seq_list_old):
            if type(new) == type(old):
                if isinstance(new, nn.Conv2d):
                    new.weight = old.weight
                if isinstance(new, nn.BatchNorm2d):
                    new.bias = old.weight
                if isinstance(new, nn.ConvTranspose2d):
                    nn.init.xavier_uniform_(new.weight)


class UNet_vgg(BaseModel):

    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1,
                 initial_features: int = 64,
                 activation_func: Callable = nn.Sigmoid):
        super(UNet_vgg, self).__init__(in_channels=in_channels,
                                       out_channels=out_channels)

        vgg_backbone = torch.nn.Sequential(*list(model.vgg16_bn(weights='DEFAULT').children())[:-2])

        self.features = initial_features

        # build encoder
        self.encoder1 = vgg_backbone[0][0:6]

        self.encoder2 = vgg_backbone[0][6:13]

        self.encoder3 = vgg_backbone[0][13:20]

        self.encoder4 = vgg_backbone[0][23:30]

        # build bottleneck
        self.bottleneck = self.random_init(nn.Sequential(block.merge(block.pool2d(),
                                                    block.sequential_block_unet(self.features * 8, self.features * 16),
                                                    block.up_sample(self.features * 16, self.features * 8))))
        # build decoder
        self.decoder4 = self.random_init(nn.Sequential(block.merge(block.sequential_block_unet(self.features * 16, self.features * 8),
                                                  block.up_sample(self.features * 8, self.features * 4))))

        self.decoder3 = self.random_init(nn.Sequential(block.merge(block.sequential_block_unet(self.features * 8, self.features * 4),
                                                  block.up_sample(self.features * 4, self.features * 2))))

        self.decoder2 = self.random_init(nn.Sequential(block.merge(block.sequential_block_unet(self.features * 4, self.features * 2),
                                                  block.up_sample(self.features * 2, self.features))))

        self.decoder1 = self.random_init(nn.Sequential(block.sequential_block_unet(self.features * 2, self.features)))
        # output layer
        self.final = self.random_init(nn.Sequential(block.merge(block.conv1x1(self.features, self.out_channels, bias=False),
                                               block.final_layer(activation_func))))

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

    def random_init(self, sequential: nn.Sequential, types: Tuple = (nn.Conv2d, nn.ConvTranspose2d)):

        if not isinstance(sequential, nn.Sequential):
            raise TypeError(f'sequential must be of type nn.Sequential. Got {type(nn.Sequential).__name__}')

        for layer in sequential.children():

            if isinstance(layer, types):
                nn.init.xavier_uniform_(layer.weight)

        return sequential


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

    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1,
                 initial_features: int = 32):
        super(ResUNet, self).__init__(in_channels=in_channels,
                                      out_channels=out_channels)

        self.encoder1 = nn.Sequential()

    def forward(self, x: Tensor) -> Tensor:
        pass


class UnetPlussPluss(BaseModel):

    def __init__(self, in_channels: int = 3,
                 out_channels: int = 1,
                 initial_features: int = 64):

        super(UnetPlussPluss, self).__init__(in_channels=in_channels,
                                             out_channels=out_channels)

    def forward(self, x: Tensor) -> Tensor:
        pass


if __name__ == '__main__':

    #from models.utils import change_weights_inputsize
    from torch.optim import Adam
    from torch.nn import Conv2d, BatchNorm2d, ConvTranspose2d, ReLU
    from unet_plusspluss.unet_plusspluss import Generic_UNetPlusPlus
    torch.manual_seed(666)
    #model = torch.load('C:\\Users\\dombe\\PycharmProjects\\Test\\weights\\mateuszbuda-brain-segmentation-pytorch.pt')
    #print(model['model_state'].keys())
    #unet = UNet(3, 1, 32)
    #unet.init_weights(model['model_state'])
    #print(model['model_state'])
    unet_plusspluss = Generic_UNetPlusPlus(3, 64, 1, 4, convolutional_pooling=False,
                                           convolutional_upsampling=True,
                                           deep_supervision=True,
                                           init_encoder='vgg')

    #print(unet_plusspluss(x))
    #res = unet_plusspluss(x)

    #res = torch.div((res[0] + res[1] + res[2] + res[3]), len(res))
    #print(res.shape)

    #for i in unet_plusspluss(x):
    #    print(i.shape)
    #print(unet_plusspluss)
    #print(unet_plusspluss)
    """
    unet = UNet(3, 1, 32)
    
    seq_list_old = []
    
    for i in seq:
        if isinstance(i, nn.Sequential):
            for child in i.children():
                seq_list_old.append(child)
        elif isinstance(i, (ConvTranspose2d, nn.MaxPool2d)):
            seq_list_old.append(i)

    seq_list_new = []
    for i in unet.children():
        if isinstance(i, nn.Sequential):
            for child in i.children():
                seq_list_new.append(child)
        elif isinstance(i, (ConvTranspose2d, nn.MaxPool2d)):
            seq_list_new.append(i)

    seq_list_new = seq_list_new[:-2]

    print(seq_list_new)
    print(list(unet.children())[0][0].weight)
    for new, old in zip(seq_list_new, seq_list_old):
        if type(new) == type(old):
            if isinstance(new, nn.Conv2d):
                new.weight = old.weight
            if isinstance(new, nn.BatchNorm2d):
                new.bias = old.weight
            if isinstance(new, ConvTranspose2d):
                nn.init.xavier_uniform_(new.weight)
    """
    #print(seq)
    #conv = torch.nn.Conv2d(3, 32, kernel_size=(3,3), stride=(1, 1), padding=(1, 1), bias=True)
    #conv.weight = seq
    #print(conv.weight)
    """
    #print(unet.encoder1.Conv1.weight.shape)

    # params = [p for p in model.parameters() if p.requires_grad]

    # torchsummary.summary(model.cuda(), (3, 64, 64))

    #print(model.encoder1.enc1conv1.weight.shape)

    #state_dict = change_weights_inputsize(model, 0
    #print(list(unet.encoder1))

    layer_names = []
    for idx, (name, param) in enumerate(unet.named_parameters()):
        layer_names.append(name)

    lr = 1e-6
    lr_mult = 10

    # placeholder
    parameters = []

    # store params & learning rates
    for idx, name in enumerate(layer_names):
        # display info
        print(f'{idx}: lr = {lr:.6f}, {name}')

        # append layer parameters
        parameters += [{'params': [p for n, p in unet.named_parameters() if n == name and p.requires_grad],
                        'lr': lr}]

        # update learning rate
        lr *= lr_mult

    #optim = Adam(parameters)
    #print(optim)
    #torchsummary.summary(unet.cuda(), (1, 64, 64))
    """
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
