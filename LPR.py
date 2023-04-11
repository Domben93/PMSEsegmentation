import os
from functools import partial
from typing import Any

import torch
from torch import Tensor
from zennit.composites import EpsilonGammaBox, SpecialFirstLayerMapComposite, layer_map_base
from zennit.rules import Gamma, Epsilon, ZBox, ZPlus, AlphaBeta, Flat, Pass, Norm
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient
from zennit.types import Convolution, ConvolutionTranspose
from matplotlib import pyplot as plt
import numpy as np
from utils.dataset import get_dataloader
import utils.transforms as t
from torch.utils.data import DataLoader
from models.unet_plusspluss.unet_plusspluss import Generic_UNetPlusPlus
from models.unets import UNet, UNet_vgg, UNet_unet
from zennit.image import imsave
import torch.nn as nn
from utils.utils import remove_from_dataname
from matplotlib import cm


def pass_func(input):
    return input


class EmptyActivation(nn.Module):
    def __init__(self):
        super(EmptyActivation, self).__init__()

    def forward(self, input):
        return input


class LPR:

    def __init__(self, model: str,
                 model_weight_path: str,
                 dataloader: DataLoader = None,
                 save_path: str = None,
                 init_features: int = 64,
                 global_relevance_max: bool = True):

        if model == 'unetplusspluss':
            self.model = Generic_UNetPlusPlus(input_channels=3,
                                              base_num_features=init_features,
                                              num_classes=1,
                                              num_pool=4,
                                              convolutional_pooling=False,
                                              convolutional_upsampling=True,
                                              deep_supervision=False,
                                              init_encoder=None,
                                              seg_output_use_bias=True,
                                              final_nonlin=pass_func)
        elif model == 'Unet':
            self.model = UNet(in_channels=3,
                              out_channels=1,
                              initial_features=32,
                              activation_func=EmptyActivation)

        elif model == 'Unet_vgg':

            self.model = UNet_vgg(in_channels=3,
                                  out_channels=1,
                                  initial_features=64,
                                  activation_func=EmptyActivation)

        elif model == 'Unet_unet':
            self.model = UNet_unet(in_channels=3,
                                   out_channels=1,
                                   initial_features=32,
                                   activation_func=EmptyActivation)
        else:
            raise ValueError(f'Model name is not supported')

        pre_trained = torch.load(model_weight_path)
        self.model.load_state_dict(pre_trained['model_state'])
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        canonizers = [SequentialMergeBatchNorm()]
        self.low = torch.zeros((1, 3, 1, 1))
        self.high = torch.ones((1, 3, 1, 1))
        self.composites = EpsilonGammaBox(low=-3.0, high=3.0, canonizers=canonizers, gamma=0.25)
        self.loader = dataloader
        self.save_path = save_path
        self.global_relevance_max = global_relevance_max

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.undo_scaling = t.UndoQuasiResize(t.QuasiResize([64, 64], 2))

    def __call__(self):

        assemble = []
        im_names = []
        for data in self.loader:
            image, mask, info = data
            image.requires_grad = True

            with Gradient(model=self.model, composite=self.composites) as attr:

                handles = [module.register_forward_hook(self.store_hook) for module in self.model.modules()]
                mask = torch.ones(mask.shape)
                output_relevance = partial(self.attr_output_fn, target=mask)
                out, relevance = attr(image, output_relevance)

                relevance = np.array(relevance[:, 0, :, :].detach().cpu())

                (lr, rr), (lc, rc) = info['split_info']
                relevance = self.undo_scaling(torch.from_numpy(relevance), (rr - lr, rc - lc))[0, :, :]
                im_name = remove_from_dataname(info['image_name'])[0]
                if im_name not in im_names:
                    im_names.append(im_name)
                assemble.append([relevance, im_name])
            for handle in handles:
                handle.remove()

        if self.global_relevance_max:
            tot_amax = 0
            for rel, name in assemble:
                rel = np.array(rel)
                max_ = np.abs(rel).max((0, 1), keepdims=True)
                if tot_amax < max_:
                    tot_amax = max_

        for name in im_names:

            name_indices = [i for i, _ in enumerate(assemble) if assemble[i][1] == name]
            cat_relevance = torch.cat([assemble[i][0] for i in name_indices], dim=1)

            if self.global_relevance_max:
                relevance = (cat_relevance + tot_amax) / 2 / tot_amax
            else:
                amax = np.abs(cat_relevance).max((0, 1), keepdims=True)
                relevance = (cat_relevance + amax) / 2 / amax

            imsave(os.path.join(self.save_path, name + f'.png'),
                   relevance, vmin=0, vmax=1, cmap='coldnhot')

    @staticmethod
    def attr_output_fn(output, target):
        return output * target

    @staticmethod
    def store_hook(module, input, output):
        # set the current module's attribute 'output' to the its tensor
        module.output = output
        # keep the output tensor gradient, even if it is not a leaf-tensor
        output.retain_grad()

    @staticmethod
    def relevance_norm(relevance: np.ndarray) -> Tensor:

        amax = np.abs(relevance).max((1, 2), keepdims=True)
        relevance = (relevance + amax) / 2 / amax

        return torch.from_numpy(relevance)


if __name__ == '__main__':
    """
    data = torch.randn(1, 3, 224, 224)
    model = vgg16_bn()

    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonGammaBox(low=-3., high=3., canonizers=canonizers)

    with Gradient(model=model, composite=composite) as attributor:
        out, relevance = attributor(data, torch.eye(1000)[[0]])

        relevance = np.array(relevance.sum(1).detach().cpu())

    print(relevance.shape)
    fig, ax = plt.subplots(2, 1)
    ax[0].imshow(np.array(data.squeeze()).transpose((1, 2, 0)))
    ax[1].imshow(relevance[0, :, :])
    plt.show()
    """
    config_path = 'models/options/unet_config.ymal'

    pair_compose = t.PairCompose([
        [t.ConvertDtype(torch.float32), t.ConvertDtype(torch.float32)],
        [t.Normalize((0, 1), (0, 255), return_type=torch.float32), None],
        [t.QuasiResize([64, 64], 2),
         t.QuasiResize([64, 64], 2)]
    ])

    loader = get_dataloader(config_path, transforms=pair_compose, mode='test')

    lpr = LPR(model='Unet_vgg',
              model_weight_path='/weights/Unet_vgg_64_pretrain-False_loss-BinaryDiceLoss_optim-adam_aug-none/lr_0.003_wd_0.007_betas_0.9-0.999_momentum_0.9_freezed-None_0.pt',
              dataloader=loader,
              save_path='/results/images/relevance_map/unet64-pretrained/',
              init_features=64,
              global_relevance_max=True)

    lpr()
