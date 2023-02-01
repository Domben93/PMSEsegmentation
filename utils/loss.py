import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import Dice

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


class BinaryDiceLoss(nn.Module):
    
    def __init__(self,
                 threshold: float = 0.5,
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

        #pred = torch.where(predict >= 0.5, 1, 0)
        #pred = predict.view(predict.shape[0], -1)
        #targ = target.view(target.shape[0], -1)

        #intersect = torch.sum(torch.mul(pred, targ), dim=1) + self.eps

        #numerator = (2. * intersect)
        #denominator = torch.sum(pred, dim=1) + torch.sum(targ, dim=1) + intersect
        #self.dice.update(predict, targ)
        #return (1 - (numerator / (denominator + self.eps))).mean()

        predict = predict.view(-1)
        target = target.view(-1)

        intersection = (predict * target).sum()
        dice = (2.0 * intersection + self.eps) / (target.sum() + predict.sum() + self.eps)

        return 1 - dice


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
        focal_loss = self.alpha * (1 - bce_exp)**self.gamma * bce

        return focal_loss


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


class DiceCoefficient(nn.Module):

    def __init__(self, smooth=1):
        super(DiceCoefficient, self).__init__()
        self.smooth = smooth

    def forward(self, pred, label):
        if pred.shape != label.shape:
            raise ValueError(f'predicted and label mus have same shape.')

        pred = pred.view(pred.shape[0], -1)
        label = label.view(label.shape[0], -1)

        intersection = (pred * label).sum(dim=1)
        union = (pred.sum(dim=1) + label.sum(dim=1))

        return (2 * intersection + self.smooth) / (union + self.smooth)


if __name__ == '__main__':

    torch.manual_seed(42)

    label = torch.rand(10, 1, 50, 50)
    pred_raw = torch.rand(10, 1, 50, 50)

    label1 = torch.tensor([[[[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1]]]])

    pred1 = torch.tensor([[[[0.0100, 0.0300, 0.0200, 0.0200, 0.0300],
          [0.0900, 0.1200, 0.0700, 0.0700, 0.0500],
          [0.0900, 0.0700, 0.0800, 0.0500, 0.0200],
          [0.9900, 0.8500, 0.9800, 0.9700, 0.9500],
          [0.8900, 0.9700, 0.8800, 0.9500, 0.8800]],
                           ]])

    loss = WeightedSoftDiceLoss(v1=0.15)
    dice = DiceCoefficient()

    label = (label >= .5).float()

    print(loss(pred1, label1))
    print(dice((pred1 >= .5).float(), label1))





