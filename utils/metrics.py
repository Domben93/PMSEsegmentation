import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from math import isnan


class mIoU(nn.Module):
    """
    Mean Intersection over Union for binary class
    calculates the IoU of batch by calling forward (__call__). To get mean IoU
    one can call compute after all batches has gone through the model pipeline.
    NB! remember to call reset if class is to be used over more than one epoch.
    Can also return loss if specified in compute when getting batch loss; call
    reset after each batch if doing so.
    """

    def __init__(self, eps=1e-4, threshold=0.5):
        super(mIoU, self).__init__()
        self.batch_count = 0
        self.iou = 0
        self.epsilon = eps
        self.threshold = threshold

    def forward(self, predicted: torch.Tensor, label: torch.Tensor):
        """
        calculate the IoU of the batch when called
        Args:
            predicted: Predicted tensor [B, H, W] or [B, 1, H, W]
            label: label tensor [B, H, W] or [B, 1, H, W]

        Stores the IoU of batch internally in class variables

        """
        if predicted.shape != label.shape:
            raise ValueError(f'shape of predicted tensor and label must be of same shape. Got {predicted.shape} and '
                             f'{label.shape} respectively')

        predicted = (predicted.view(-1) >= self.threshold).float()
        label = label.view(-1)

        intersection = torch.sum((predicted * label)).item()
        union = torch.sum((predicted + label)).item() - intersection

        self.iou += intersection / union if union > 0 else 1
        self.batch_count += 1

    def compute(self, loss=False):
        """
        calculates the mean IoU of all the data that has been given
        by dividing by the number of batches i.e. number of times the forward
        method has been called
        Args:
            loss:

        Returns:

        """
        if loss:
            return 1 - (self.iou / self.batch_count)
        else:
            return self.iou / self.batch_count

    def reset(self):
        self.iou = 0
        self.batch_count = 0


class DiceCoefficient(nn.Module):

    def __init__(self, smooth=1):
        super(DiceCoefficient, self).__init__()
        self.smooth = smooth

    def forward(self, pred, label):
        if pred.shape != label.shape:
            raise ValueError(f'')

        pred = pred.view(pred.shape[0], -1)
        label = label.view(label.shape[0], -1)

        intersection = (pred * label).sum(dim=1)
        union = (pred.sum(dim=1) + label.sum(dim=1))

        return (2 * intersection + self.smooth) / (union + self.smooth)


class BinaryMetrics(nn.Module):

    def __init__(self):
        super(BinaryMetrics, self).__init__()

        self.batch_count = 0
        self.batch_pred = []
        self.batch_true = []
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def forward(self, pred: Tensor, true: Tensor):

        if pred.shape != true.shape:
            raise RuntimeError(f"predicted Tensor Shape {pred.shape} != label Tensor Shape {true.shape}")

        for i in range(pred.shape[0]):
            self.batch_pred.append(pred[i].view(-1).numpy())
            self.batch_true.append(true[i].view(-1).numpy())

        self.batch_count += pred.shape[0]

    def reset(self):
        self.batch_count = 0
        self.batch_pred = []
        self.batch_true = []
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

    def confusion_matrix(self, threshold=0.5):
        cm = 0
        for prediction, true in zip(self.batch_pred, self.batch_true):
            cm = cm + confusion_matrix(true, np.where(prediction >= threshold, 1, 0))

        self.TN = cm[0, 0]
        self.FN = cm[1, 0]
        self.FP = cm[0, 1]
        self.TP = cm[1, 1]

        return cm

    def FNR(self) -> float:
        return self.FN / (self.FN + self.TP)

    def FPR(self) -> float:
        return self.FP / (self.FP + self.TN)

    def accuracy(self) -> float:
        """
        Returns:
        """
        return (self.TP + self.TN) / (self.TP + self.TN + self.FN + self.FP)

    def mIoU(self):
        return self.TP / (self.TP + self.FN + self.FP)

"""
    def dice_coefficient(self, as_loss: bool = False) -> float:

        dsc = 2 *
        pass
        
"""
