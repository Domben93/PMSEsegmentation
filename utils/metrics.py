import math
from typing import List, NoReturn, Any, Tuple, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.functional.classification import *


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

    def forward(self, predicted: torch.Tensor, label: torch.Tensor, data_info: List):
        """

        Args:
            predicted: Predicted tensor [B, H, W] or [B, 1, H, W]
            label: label tensor [B, H, W] or [B, 1, H, W]
            data_info: Any pickle

        """

        if predicted.shape != label.shape:
            raise ValueError(f'shape of predicted tensor and label must be of same shape. Got {predicted.shape} and '
                             f'{label.shape} respectively')

        if isinstance(data_info, str):
            data_info = [data_info]

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

    def mIoU(self, threshold: float = 0.5, ignore_index: int = None, sample_classes: List[List[str]] = None) -> Tensor:
        """
        calculates the mean IoU of all the data that has been given
        by dividing by the number of batches i.e. number of times the forward
        method has been called
        Args:
            sample_classes:
            threshold:
            ignore_index:
        Returns:
        """
        return self.__compute(binary_jaccard_index, sample_classes, ignore_index=ignore_index, threshold=threshold)

    def auc(self, threshold: float = None, ignore_index: int = None, sample_classes: List[List[str]] = None) -> Tensor:
        """

        Args:
            threshold:
            ignore_index:
            sample_classes:

        Returns:

        """
        return self.__compute(binary_auroc, sample_classes, ignore_index=ignore_index, thresholds=threshold)

    def accuracy(self, threshold: float = 0.5, ignore_index: int = None,
                 sample_classes: List[List[str]] = None) -> Tensor:
        """

        Args:
            threshold:
            ignore_index:
            sample_classes:

        Returns:

        """
        return self.__compute(binary_accuracy, sample_classes, threshold=threshold, ignore_index=ignore_index)

    def precision(self, threshold: float = 0.5, ignore_index: int = None,
                  sample_classes: List[List[str]] = None) -> Tensor:
        """

        Args:
            threshold:
            ignore_index:
            sample_classes:

        Returns:

        """
        return self.__compute(binary_precision, sample_classes, threshold=threshold, ignore_index=ignore_index)

    def dice(self, threshold: float = 0.5, ignore_index: int = None, sample_classes: List[List[str]] = None) -> Tensor:
        """

        Args:
            threshold:
            ignore_index:
            sample_classes:

        Returns:

        """
        return self.__compute(binary_f1_score, sample_classes, threshold=threshold, ignore_index=ignore_index)

    def confusion_matrix(self, threshold: float = 0.5, ignore_index: int = None,
                         sample_classes: List[List[str]] = None) -> Tensor:
        """

        Args:
            threshold:
            ignore_index:
            sample_classes:

        Returns:

        """
        return self.__compute(binary_confusion_matrix, sample_classes, threshold=threshold, ignore_index=ignore_index)

    def FPR_FNR(self, threshold: float = 0.5, ignore_index: int = None, sample_classes: List[List[str]] = None)\
            -> Tensor:
        """

        Args:
            threshold:
            ignore_index:
            sample_classes:

        Returns:

        """
        confusion_mats = self.__compute(binary_confusion_matrix, sample_classes,
                                        threshold=threshold,
                                        ignore_index=ignore_index)

        fnr_fpr = torch.zeros(2, (len(sample_classes)))

        for idx, conf_mat in enumerate(confusion_mats):

            fnr_fpr[0, idx] = conf_mat[0, 1] / (conf_mat[0, 1] + conf_mat[0, 0])
            fnr_fpr[1, idx] = conf_mat[1, 0] / (conf_mat[1, 0] + conf_mat[1, 1])

        return fnr_fpr

    def __compute(self, function: Callable, sample_classes: List[List[str]] = None, **kwargs) -> Any:

        self.__check_data_validity()

        num_different_samples = len(list(self._store_res.keys()))

        if sample_classes:
            total = {}
            for i in range(len(sample_classes)):
                total[i] = []
        else:
            total = 0

        for key in self._store_res.keys():

            partial_met_val = function(self._store_res[key]['predicted'], self._store_res[key]['label'],
                                       **kwargs)

            if torch.all(torch.isnan(partial_met_val)):
                partial_met_val = 1

            if sample_classes is not None:

                index = self.__find_in_nested_list(sample_classes, key)
                sample_class = index[0]

                total[sample_class].append(partial_met_val)

            else:
                total += partial_met_val

        if isinstance(total, dict):
            res = []

            for key in total.keys():

                if function is binary_confusion_matrix:

                    res.append(sum(total[key]))
                else:
                    res.append(float(sum(total[key]) / len(total[key])))
        else:
            res = total / num_different_samples

        return res

    def __check_data_validity(self):
        if not bool(self._store_res):
            raise RuntimeError('Tried to compute metric with no available data. '
                               'Use class call with the prediction, label and info to store data.')

    @staticmethod
    def __find_in_nested_list(nested_list: List, string: str) -> Tuple:

        for sub_list in nested_list:
            if string in sub_list:
                return nested_list.index(sub_list), sub_list.index(string)
        raise ValueError("'{char}' is not in list".format(char=string))


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
