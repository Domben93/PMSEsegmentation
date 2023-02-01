import math
from typing import List, NoReturn

import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.functional.classification import *
from math import isnan


class SegMets(nn.Module):
    """
    Mean Intersection over Union for binary class
    calculates the IoU of batch by calling forward (__call__). To get mean IoU
    one can call compute after all batches has gone through the model pipeline.
    NB! remember to call reset if class is to be used over more than one epoch.
    Can also return loss if specified in compute when getting batch loss; call
    reset after each batch if doing so.
    """

    def __init__(self):
        super(SegMets, self).__init__()

        self._store_res = {}

    def forward(self, predicted: torch.Tensor, label: torch.Tensor, data_info: list):
        """

        Args:
            predicted: Predicted tensor [B, H, W] or [B, 1, H, W]
            label: label tensor [B, H, W] or [B, 1, H, W]
            data_info: Any pickle

        """
        if predicted.shape != label.shape:
            raise ValueError(f'shape of predicted tensor and label must be of same shape. Got {predicted.shape} and '
                             f'{label.shape} respectively')

        predicted = predicted.view(predicted.shape[0], -1)
        label = label.view(label.shape[0], -1)

        for num, im in enumerate(data_info):
            if im in self._store_res:
                self._store_res[im]['predicted'] = torch.cat(
                    (predicted[num, :].unsqueeze(dim=0), self._store_res[im]['predicted']))
                self._store_res[im]['label'] = torch.cat((label[num, :].unsqueeze(dim=0), self._store_res[im]['label']))
            else:
                self._store_res[im] = {}
                self._store_res[im]['predicted'] = predicted[num, :].unsqueeze(dim=0)
                self._store_res[im]['label'] = label[num, :].unsqueeze(dim=0)

    def mIoU(self, threshold: float = 0.5, ignore_index: int = None) -> float:
        """
        calculates the mean IoU of all the data that has been given
        by dividing by the number of batches i.e. number of times the forward
        method has been called
        Args:
            threshold:
            ignore_index:
        Returns:
        """

        self.__check_storage()

        num_imgs = len(list(self._store_res.keys()))
        tot_jacc = 0

        for key in self._store_res.keys():

            jacc = binary_jaccard_index(self._store_res[key]['predicted'], self._store_res[key]['label'],
                                        threshold=threshold,
                                        ignore_index=ignore_index).item()

            if math.isnan(jacc):
                jacc = 1.0

            tot_jacc += jacc

        tot_jacc = tot_jacc / num_imgs

        if math.isnan(tot_jacc):
            tot_jacc = 0.0

        return tot_jacc

    def auc(self) -> float:
        """

        Returns:

        """
        num_imgs = len(list(self._store_res.keys()))
        tot_auc = 0

        for key in self._store_res.keys():
            auc = binary_auroc(self._store_res[key]['predicted'], self._store_res[key]['label'])

            if math.isnan(auc):
                raise ValueError('Nan value detected')

            tot_auc += auc

        return tot_auc / num_imgs

    def accuracy(self, threshold: float = 0.5, ignore_index: int = None):
        """

        Returns:

        """
        num_imgs = len(list(self._store_res.keys()))
        tot_acc = 0

        for key in self._store_res.keys():
            acc = binary_accuracy(self._store_res[key]['predicted'], self._store_res[key]['label'],
                                  threshold=threshold,
                                  ignore_index=ignore_index)

            if math.isnan(acc):
                raise ValueError('Nan value detected')

            tot_acc += acc

        return tot_acc / num_imgs

    def precision(self, threshold: float = 0.5, ignore_index: int = None):

        num_imgs = len(list(self._store_res.keys()))
        tot_prec = 0

        for key in self._store_res.keys():
            prec = binary_precision(self._store_res[key]['predicted'], self._store_res[key]['label'],
                                    threshold=threshold,
                                    ignore_index=ignore_index)

            if math.isnan(prec):
                raise ValueError('Nan value detected')

            tot_prec += prec

        return tot_prec / num_imgs

    def dice(self, threshold: float = 0.5, ignore_index: int = None):

        num_imgs = len(list(self._store_res.keys()))
        tot_dice = 0

        for key in self._store_res.keys():
            dice = binary_f1_score(self._store_res[key]['predicted'], self._store_res[key]['label'],
                                   threshold=threshold,
                                   ignore_index=ignore_index)

            if math.isnan(dice):
                raise ValueError('Nan value detected')

            tot_dice += dice

        return tot_dice / num_imgs

    def reset(self):
        self._store_res = {}

    def __check_storage(self):
        if not bool(self._store_res):
            raise RuntimeError(f'No values available for computation.')


class mIoU(nn.Module):
    """
    Mean Intersection over Union for binary class
    calculates the IoU of batch by calling forward (__call__). To get mean IoU
    one can call compute after all batches has gone through the model pipeline.
    NB! remember to call reset if class is to be used over more than one epoch.
    Can also return loss if specified in compute when getting batch loss; call
    reset after each batch if doing so.
    """

    def __init__(self, eps=1e-4, threshold=0.5, reset_after_compute=False, device=None):
        super(mIoU, self).__init__()

        self.epsilon = eps
        self.automatic_reset = reset_after_compute
        self.batch_count = 0
        self.iou = {}
        self.jaccard = BinaryJaccardIndex(threshold=threshold)
        self.device = device

        if self.device:
            self.jaccard.cuda(device)

    def forward(self, predicted: torch.Tensor, label: torch.Tensor, data_info: list):
        """
        calculate the mean IoU for image wise
        Args:
            predicted: Predicted tensor [B, H, W] or [B, 1, H, W]
            label: label tensor [B, H, W] or [B, 1, H, W]
            data_info:
        Stores the IoU of batch internally in class variables

        """
        if predicted.shape != label.shape:
            raise ValueError(f'shape of predicted tensor and label must be of same shape. Got {predicted.shape} and '
                             f'{label.shape} respectively')

        predicted = predicted.view(predicted.shape[0], -1)
        label = label.view(label.shape[0], -1)

        if self.device:
            if not (predicted.is_cuda and label.is_cuda):
                raise RuntimeError('The predicted Tensor and the Mask Tensor must both be on the same device'
                                   ' as metric class.')

        for num, im in enumerate(data_info):
            if im in self.iou:
                self.iou[im]['predicted'] = torch.cat((predicted[num, :].unsqueeze(dim=0), self.iou[im]['predicted']))
                self.iou[im]['label'] = torch.cat((label[num, :].unsqueeze(dim=0), self.iou[im]['label']))
            else:
                self.iou[im] = {}
                self.iou[im]['predicted'] = predicted[num, :].unsqueeze(dim=0)
                self.iou[im]['label'] = label[num, :].unsqueeze(dim=0)

    def compute(self, loss=False) -> float:
        """
        calculates the mean IoU of all the data that has been given
        by dividing by the number of batches i.e. number of times the forward
        method has been called
        Args:
            loss:

        Returns:

        """
        """
        if self.batch_union == 0:
            raise ZeroDivisionError('Union is equal to zero. Division by Zero is not allowed.')

        iou = self.batch_intersection / self.batch_union

        if loss:
            iou = 1 - iou

        if self.automatic_reset:
            self.reset()
        """
        num_imgs = len(list(self.iou.keys()))
        tot_jacc = 0

        for key in self.iou.keys():

            jacc = self.jaccard(self.iou[key]['predicted'], self.iou[key]['label']).item()

            if math.isnan(jacc):
                jacc = 1.0

            tot_jacc += jacc

        tot_jacc = tot_jacc / num_imgs

        if math.isnan(tot_jacc):
            tot_jacc = 0.0

        if self.automatic_reset:
            self.reset()

        return tot_jacc

    def reset(self):
        self.iou = {}


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
