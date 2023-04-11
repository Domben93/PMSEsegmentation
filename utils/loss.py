from functools import partial
from operator import itemgetter
from typing import Tuple, Union, List, Callable
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor, einsum
from torch.nn import functional as F
from torchvision import transforms
from scipy.ndimage import distance_transform_edt as eucl_distance

D = Union[Image.Image, np.ndarray, Tensor]


class BinaryDiceLoss(nn.Module):

    def __init__(self,
                 threshold: float = None,
                 eps: float = 1e-7,
                 reduction: str = 'mean'):
        super(BinaryDiceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, predict: Tensor, target: Tensor):

        if predict.shape[0] != target.shape[0]:
            raise ValueError(f'Predicted Tensor must have same shape as target Tensor: '
                             f'Predicted shape ({predict.shape[0]}) != ({target.shape[0]}) Target shape')
        if predict.shape[1] != 1 or target.shape[1] != 1:
            raise ValueError(f'Predicted and target Tensor must have shape [N, 1, H, W]. Got predicted: {predict.shape}'
                             f' and target: {target.shape}')

        predict = predict.view(predict.shape[0], -1)
        target = target.view(target.shape[0], -1)

        intersection = (predict * target).sum(dim=1)

        dice = (2.0 * intersection + self.eps) / (target.sum(dim=1) + predict.sum(dim=1) + self.eps)

        return (1 - dice).mean()


class GeneralizedDiceLoss(nn.Module):

    def __init__(self, eps: float = 1e-10):
        super(GeneralizedDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:

        pred = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)

        w_f = 1 / ((torch.sum(target, dim=1) + self.eps)**2)
        w_b = 1 / ((torch.sum(1 - target, dim=1) + self.eps)**2)

        fg_intersect = (w_f * (pred * target).sum(dim=1))
        fb_intersect = (w_b * ((1 - pred) * (1 - target)).sum(dim=1))

        fg_tot = (w_f * (pred + target).sum(dim=1))
        fb_tot = (w_b * (2 - pred - target).sum(dim=1))

        gdl = 1 - (2 * ((fg_intersect + fb_intersect + self.eps) / (fg_tot + fb_tot + self.eps)))

        return gdl.mean()


# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2, alpha: float = 0.8, output: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.output = output

    def forward(self, predicted: Tensor, label: Tensor) -> Tensor:
        predicted = predicted.view(-1)
        label = label.view(-1)

        bce = F.binary_cross_entropy(predicted, label, reduction=self.output)

        bce_exp = torch.exp(-bce)
        focal_loss = self.alpha * (1 - bce_exp) ** self.gamma * bce

        return focal_loss.mean()


class LogCoshDiceLoss(nn.Module):

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        super(LogCoshDiceLoss, self).__init__()
        self.dice = BinaryDiceLoss(threshold=threshold, eps=eps)

    def forward(self, predicted: Tensor, label: Tensor) -> Tensor:
        loss = torch.log(torch.cosh(self.dice(predicted, label)))

        return loss


class WeightedSoftDiceLoss(nn.Module):

    def __init__(self, v1=0.1, eps=1e-6):
        super(WeightedSoftDiceLoss, self).__init__()
        self.eps = eps
        self.v1 = v1
        self.v2 = 1 - v1

        if self.v1 < 0:
            raise ValueError(f'v1 must be positive valued. Got {self.v1}')

        elif self.v1 > 0.5:
            raise ValueError(f'v1 cannot be larger than 0.5 because of 0 <= v1 <= v2 <= 1. Got {self.v1}')

    def forward(self, predicted: Tensor, label: Tensor, logits=False) -> Tensor:
        """

        Args:
            predicted: tensor of shape [N, 1, H, W] or [N, H, W] with the predicted probabilities.
            label: tensor of shape [N, 1, H, W] or [N, H, W] containing binary labels
            logits: if predicted tensor is raw (no activation function applied to it) then logits can be
                    set to True in order to use the sigmoid activation such that we get the probabilities

        Returns:

        """

        if logits:
            predicted = F.sigmoid(predicted)

        label = label.view(label.shape[0], -1)
        predicted = predicted.view(predicted.shape[0], -1)

        w = label * (self.v2 - self.v1) + self.v1

        X = w * (2 * predicted - 1)
        Y = w * (2 * label - 1)

        intersection = (X * Y).sum(dim=1)

        LWDice = (2 * intersection + self.eps) / (X.sum(dim=1) + Y.sum(dim=1) + self.eps)

        return (1 - LWDice).mean()


class DiceBCELoss(nn.Module):
    def __init__(self, weight=(1, 0.5), size_average=True):
        super(DiceBCELoss, self).__init__()
        if weight is None:
            weight = (1, 1)
        self.dice_weight = weight[0]
        self.bce_weight = weight[1]

    def forward(self, inputs, targets, smooth=1):

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        if self.dice_weight and self.bce_weight:
            Dice_BCE = self.bce_weight * BCE + self.dice_weight * dice_loss
            return Dice_BCE
        else:
            return BCE + dice_loss


class SurfaceLoss(nn.Module):

    def __init__(self, **kwargs):
        super(SurfaceLoss, self).__init__()
        self.dist_transform = self.dist_map_transform([1, 1], 1)

    def forward(self, prob: Tensor, label: Tensor):

        f_pc = prob.type(torch.float32).cpu()
        f_dc = self.dist_transform(label.cpu())

        f_multiplied = einsum("bkwh,bkwh->bkwh", f_pc, f_dc)

        loss = f_multiplied.mean()

        return loss

    @staticmethod
    def dist_map_transform(resolution: List[float], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
            SurfaceLoss.gt_transform(resolution, K),
            lambda t: t.cpu().numpy(),
            partial(SurfaceLoss.one_hot2dist, resolution=resolution),
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

    @staticmethod
    def gt_transform(resolution: List[float], K: int) -> Callable[[D], Tensor]:
        return transforms.Compose([
            lambda img: np.array(img)[...],
            lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],  # Add one dimension to simulate batch
            # partial(class2one_hot, K=K),
            itemgetter(0)  # Then pop the element to go back to img shape
        ])

    @staticmethod
    def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                     dtype=None) -> np.ndarray:

        # assert one_hot(torch.tensor(seg), axis=0)

        K: int = len(seg)

        res = np.zeros_like(seg, dtype=dtype)
        for k in range(K):

            posmask = seg[k].astype(np.bool_)[0, :, :]

            if posmask.any():
                negmask = ~posmask
                res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                 - (eucl_distance(posmask, sampling=resolution) - 1) * posmask

        return res


class DiceSurfaceLoss(nn.Module):

    def __init__(self, alpha=1, max_alpha=0.90, strategy='constant', inc_reb_rate_iter=10):
        super(DiceSurfaceLoss, self).__init__()

        self.dice = BinaryDiceLoss()
        self.surface = SurfaceLoss()
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.strategy = strategy
        if self.strategy not in ['constant', 'increase', 'rebalance']:
            raise ValueError(f'strategy must be one of the following {["constant", "increase", "rebalance"]}.'
                             f'Got {self.strategy}')

        if self.strategy in ['rebalance', 'increase']:
            self.alpha = 0
        self.iter = 0
        self.inc_reb_rate = inc_reb_rate_iter

    def forward(self, prob: Tensor, label: Tensor):
        dice = self.dice(prob, label)
        surface = self.surface(prob.cpu(), label.cpu())

        if self.strategy == 'constant':
            loss = dice + self.alpha * surface

        elif self.strategy == 'increase':
            if self.iter % self.inc_reb_rate == 0:
                self.alpha += 0.01
            if self.alpha >= self.max_alpha:
                self.alpha = self.max_alpha
            loss = dice + self.alpha * surface
            self.iter += 1
        else:
            if self.iter % self.inc_reb_rate == 0:
                if self.iter < 100:
                    self.alpha += 0.005
                elif 100 <= self.iter <= 300:
                    self.alpha += 0.01
                else:
                    self.alpha += 0.02
            if self.alpha >= self.max_alpha:
                self.alpha = self.max_alpha
            self.iter += 1
            loss = ((1 - self.alpha) * dice) + (self.alpha * surface)

        return loss


class BCE(nn.Module):

    def __init__(self, inverse_weighting: bool = True):
        super(BCE, self).__init__()
        self.weighting = inverse_weighting
        self.bce = torch.nn.BCELoss()

    def forward(self, prob: Tensor, label: Tensor) -> Tensor:
        return self.bce(prob, label)


class BCESurfaceLoss(nn.Module):

    def __init__(self, alpha=1, max_alpha=1, strategy='constant', inc_reb_rate_iter=10,):
        super(BCESurfaceLoss, self).__init__()

        self.bce = torch.nn.BCELoss()
        self.surface = SurfaceLoss()
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.strategy = strategy
        if self.strategy not in ['constant', 'increase', 'rebalance']:
            raise ValueError(f'strategy must be one of the following {["constant", "increase", "rebalance"]}.'
                             f'Got {self.strategy}')

        if self.strategy in ['rebalance', 'increase']:
            self.alpha = 0
        self.iter = 0
        self.inc_reb_rate = inc_reb_rate_iter

    def forward(self, prob: Tensor, label: Tensor):
        bce = self.bce(prob, label)
        surface = self.surface(prob.cpu(), label.cpu())

        if self.strategy == 'constant':
            loss = bce + self.alpha * surface

        elif self.strategy == 'increase':
            if (self.iter % self.inc_reb_rate == 0) and self.max_alpha > self.alpha:
                self.alpha += 0.01
            loss = bce + self.alpha * surface
            self.iter += 1
        else:
            if (self.iter % self.inc_reb_rate == 0) and self.max_alpha > self.alpha:
                self.alpha += 0.01
            self.iter += 1
            loss = (1 - self.alpha) * bce + self.alpha * surface

        return loss


class FocalSurfaceLoss(nn.Module):

    def __init__(self, alpha=1, max_alpha=1, strategy='constant', inc_reb_rate_iter=10,):
        super(FocalSurfaceLoss, self).__init__()

        self.focal = FocalLoss()
        self.surface = SurfaceLoss()
        self.alpha = alpha
        self.max_alpha = max_alpha
        self.strategy = strategy
        if self.strategy not in ['constant', 'increase', 'rebalance']:
            raise ValueError(f'strategy must be one of the following {["constant", "increase", "rebalance"]}.'
                             f'Got {self.strategy}')

        if self.strategy in ['rebalance', 'increase']:
            self.alpha = 0
        self.iter = 0
        self.inc_reb_rate = inc_reb_rate_iter

    def forward(self, prob: Tensor, label: Tensor):
        focal = self.focal(prob, label)
        surface = self.surface(prob.cpu(), label.cpu())

        if self.strategy == 'constant':
            loss = focal + self.alpha * surface

        elif self.strategy == 'increase':
            if (self.iter % self.inc_reb_rate == 0) and self.max_alpha > self.alpha:
                self.alpha += 0.01
            loss = focal + self.alpha * surface
            self.iter += 1
        else:
            if (self.iter % self.inc_reb_rate == 0) and self.max_alpha > self.alpha:
                self.alpha += 0.01
            self.iter += 1
            loss = (1 - self.alpha) * focal + self.alpha * surface

        return loss


class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.8, beta=0.2, eps=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, predicted: Tensor, label: Tensor) -> Tensor:

        predicted = predicted.view(-1)
        label = label.view(-1)

        TP = (predicted * label).sum()
        FP = ((1 - label) * predicted).sum()
        FN = (label * (1 - predicted)).sum()

        loss = (TP + self.eps) / (TP + self.alpha * FP + self.beta*FN + self.eps)

        return (1 - loss).mean()


if __name__ == '__main__':
    torch.manual_seed(42)

    #label = torch.rand(10, 1, 50, 50)
    pred_raw = torch.rand(10, 1, 50, 50)

    label1 = torch.tensor([[
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]],
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1]]]])

    pred1 = torch.tensor([[[[0.0001, 0.0300, 0.0200, 0.0200, 0.0300],
                            [0.0900, 0.1200, 0.0700, 0.0700, 0.0500],
                            [0.0900, 0.0700, 0.0800, 0.0500, 0.0200],
                            [0.0900, 0.0500, 0.0800, 0.0700, 0.0500],
                            [0.8900, 0.9700, 0.8800, 0.9500, 0.8800]],
                           [[0.0100, 0.0300, 0.0200, 0.0200, 0.0300],
                            [0.0900, 0.1200, 0.0700, 0.0700, 0.0500],
                            [0.0900, 0.0700, 0.0800, 0.0500, 0.0200],
                            [0.0900, 0.0500, 0.0800, 0.0700, 0.0500],
                            [0.8900, 0.9700, 0.8800, 0.9500, 0.8800]]
                           ]])
    #label1 = torch.zeros((2, 1, 5, 5))

    label1 = torch.zeros((2, 1, 5, 5))
    loss = SurfaceLoss(idc=[1])

    print(loss(pred1.view(2, 1, 5, 5), label1.view(2, 1, 5, 5)))
