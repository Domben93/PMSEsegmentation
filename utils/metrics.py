import math
from typing import List, NoReturn, Any, Tuple, Callable, Dict, Literal
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.functional.classification import *

MODE = Literal['global', 'mean', 'classwise']


class SegMets(nn.Module):

    """
    Segmentation metrics used. The class includes the metrics: IoU, AUC, Dice, Accuracy, Confusion metric,
    Precision, FPR/FNR (returns the pair).
    Use the object call to add predictions and labels from runs in form of [B, C, H, W].
    """

    def __init__(self, auto_reset=False):
        """

        Args:
            auto_reset: If auto reset is set to True the stored predictions and labels will be
            reset automatically after call of the metric methods. If False, the reset() method must be called \
            manually to avoid predictions and labels to being piled up between batches/iterations. \
            In addition, a list of the class that the different samples belong to must be supplied \
            to be able to calculate the mean or classwise of the results.
            NOTE that the object stores all the predictions and labels such that if very large batches (or reset isn't
            called) are used it will consume a lot of memory such that calling .detach().cpu() might be a good idea.
            However, it is fastest if the computation are done on GPU if available.

            EXAMPLE:

                metrics = SegMets()
                sample_info: List[List] = [[],...,[]]
                model.train()

                for data in data_loader:
                    img, mask = data
                    img = img.to(device)
                    mask = mask.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(mode=True):

                        prediction = model(img)

                        loss = loss_func(prediction, mask)

                        stored_res = metrics(prediction.detach().cpu(), mask.detach.cpu(), sample_info)

                miou = stored_res.mIoU(threshold=0.5, sample_classes=sample_info, multidim_average='mean')
                dice = stored_res.dice(threshold=0.5, sample_classes=sample_info, multidim_average='mean')
                metric.reset() # Must be called if SegMets(auto_reset=False)

                print(f'mIoU: {iou}')

        """
        super(SegMets, self).__init__()

        self._store_res = {}
        self.auto_reset = auto_reset

    def forward(self, predicted: torch.Tensor, label: torch.Tensor, data_info: List) -> NoReturn:
        """

        Args:
            predicted: Predicted tensor [B, H, W] or [B, 1, H, W]
            label: label tensor [B, H, W] or [B, 1, H, W]
            data_info: List of sample i.e., class name, sample name or any other

        """

        if predicted.shape != label.shape:
            raise ValueError(f'shape of predicted tensor and label must be of same shape. Got {predicted.shape} and '
                             f'{label.shape} respectively')

        if isinstance(data_info, str):
            data_info = [data_info]

        predicted = predicted.contiguous().view(predicted.shape[0], -1)

        label = label.contiguous().view(label.shape[0], -1)

        for num, im in enumerate(data_info):
            if im in self._store_res:
                self._store_res[im]['predicted'] = torch.cat(
                    (predicted[num, :].unsqueeze(dim=0), self._store_res[im]['predicted']))
                self._store_res[im]['label'] = torch.cat((label[num, :].unsqueeze(dim=0), self._store_res[im]['label']))
            else:
                self._store_res[im] = {}
                self._store_res[im]['predicted'] = predicted[num, :].unsqueeze(dim=0)
                self._store_res[im]['label'] = label[num, :].unsqueeze(dim=0)

    def mIoU(self, threshold: float = 0.5, ignore_index: int = None, sample_classes: List[List[str]] = None,
             multidim_average: MODE = 'mean') -> Tensor:
        """
        calculates the mean IoU of all the data that has been given
        by dividing by the number of batches i.e. number of times the forward
        method has been called
        Args:
            sample_classes:
            threshold:
            ignore_index:
            multidim_average:
        Returns:
        """
        return self.__compute(binary_jaccard_index,
                              sample_classes,
                              ignore_index=ignore_index,
                              threshold=threshold,
                              multidim_average=multidim_average)

    def auc(self, threshold: float = None, ignore_index: int = None, sample_classes: List[List[str]] = None,
            multidim_average: MODE = 'global') -> Tensor:
        """

        Args:
            threshold:
            ignore_index:
            sample_classes:
            multidim_average:

        Returns:

        """
        return self.__compute(binary_auroc,
                              sample_classes,
                              ignore_index=ignore_index,
                              thresholds=threshold,
                              multidim_average=multidim_average)

    def accuracy(self, threshold: float = 0.5, ignore_index: int = None,
                 sample_classes: List[List[str]] = None, multidim_average: MODE = 'mean') -> Tensor:
        """

        Args:
            threshold:
            ignore_index:
            sample_classes:
            multidim_average:
        Returns:

        """
        return self.__compute(binary_accuracy,
                              sample_classes,
                              threshold=threshold,
                              ignore_index=ignore_index,
                              multidim_average=multidim_average)

    def precision(self, threshold: float = 0.5, ignore_index: int = None,
                  sample_classes: List[List[str]] = None, multidim_average: MODE = 'mean') -> Tensor:
        """

        Args:
            threshold:
            ignore_index:
            sample_classes:

        Returns:

        """
        return self.__compute(binary_precision,
                              sample_classes,
                              threshold=threshold,
                              ignore_index=ignore_index,
                              multidim_average=multidim_average)

    def dice(self, threshold: float = 0.5, ignore_index: int = None, sample_classes: List[List[str]] = None,
             multidim_average: MODE = 'mean') -> Tensor:
        """

        Args:
            threshold:
            ignore_index:
            sample_classes:

        Returns:

        """
        return self.__compute(binary_f1_score,
                              sample_classes,
                              threshold=threshold,
                              ignore_index=ignore_index,
                              multidim_average=multidim_average)

    def confusion_matrix(self, threshold: float = 0.5, ignore_index: int = None,
                         sample_classes: List[List[str]] = None, multidim_average: MODE = 'global') -> Tensor:
        """

        Args:
            threshold:
            ignore_index:
            sample_classes:
        Returns:

        """
        return self.__compute(binary_confusion_matrix,
                              sample_classes,
                              threshold=threshold,
                              ignore_index=ignore_index,
                              multidim_average=multidim_average)

    def FPR_FNR(self, threshold: float = 0.5, ignore_index: int = None, sample_classes: List[List[str]] = None,
                multidim_average: MODE = 'classwise') -> Tensor:
        """
        Args:
            threshold:
            ignore_index:
            sample_classes:

        Returns:

        """
        confusion_mats = self.__compute(binary_confusion_matrix, sample_classes,
                                        threshold=threshold,
                                        ignore_index=ignore_index,
                                        multidim_average=multidim_average)
        fpr_fnr = torch.zeros(2, (len(sample_classes)))

        for idx, conf_mat in enumerate(confusion_mats):
            fpr_fnr[0, idx] = conf_mat[0, 1] / (conf_mat[0, 1] + conf_mat[0, 0])
            fpr_fnr[1, idx] = conf_mat[1, 0] / (conf_mat[1, 0] + conf_mat[1, 1])

        return fpr_fnr

    def __compute(self, function: Callable, sample_classes: List[List[str]] = None, multidim_average: MODE = 'mean',
                  **kwargs) -> Any:

        self.__check_available_data()

        sample_classes = self.__remove_nonexisting_sampleclasses(list(self._store_res.keys()), sample_classes)

        if sample_classes:
            if multidim_average == 'global':
                data = self.__merge_sampledata([self.__collapse_nested_list(sample_classes)])
            else:
                data = self.__merge_sampledata(sample_classes=sample_classes)
        else:
            data = self._store_res

        num_different_samples = len(list(data.keys()))

        if multidim_average == 'classwise' and sample_classes:
            total = {}
            for i in range(len(list(data.keys()))):
                total[i] = []
        else:
            total = 0

        for i, key in enumerate(data.keys()):

            partial_met_val = function(data[key]['predicted'], data[key]['label'],
                                       **kwargs)

            if torch.all(torch.isnan(partial_met_val)):
                partial_met_val = 1

            if multidim_average == 'classwise':

                total[i].append(partial_met_val)

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

        if self.auto_reset:
            self.reset()

        return res

    def reset(self) -> NoReturn:
        self._store_res = {}

    def __check_available_data(self) -> NoReturn:
        if not bool(self._store_res):
            raise RuntimeError('Tried to compute metric with no available data. '
                               'Use class call with the prediction, label and info to store data.')

    def __merge_sampledata(self, sample_classes: List[List[str]]) -> Dict:

        new_store_structure = {}
        for class_num, sample_names in enumerate(sample_classes):
            new_store_structure[class_num] = {}
            new_store_structure[class_num]['predicted'] = torch.cat([self._store_res[i]['predicted'].view(-1) for i in sample_names])
            new_store_structure[class_num]['label'] = torch.cat([self._store_res[i]['label'].view(-1) for i in sample_names])

        return new_store_structure

    @staticmethod
    def __check_label_validity(label: Tensor) -> bool:

        if torch.all(torch.unique(label)) == 0:
            return False
        else:
            return True

    @staticmethod
    def __find_in_nested_list(nested_list: List, string: str) -> Tuple:

        for sub_list in nested_list:
            if string in sub_list:
                return nested_list.index(sub_list), sub_list.index(string)
        raise ValueError("'{char}' is not in list".format(char=string))

    @staticmethod
    def __collapse_nested_list(list_: List) -> List:
        flat_list = []
        for sublist in list_:
            for item in sublist:
                flat_list.append(item)
        return flat_list

    @staticmethod
    def __remove_nonexisting_sampleclasses(info_list: List, sample_list_: List) -> List:

        new_sample_list = [[subelt for subelt in elt if subelt in info_list] for elt in sample_list_]
        new_sample_list = [x for x in new_sample_list if x != []]

        return new_sample_list


class CEV:

    def __init__(self, baseline: Tensor = None):
        if baseline:
            self.baseline_model_pair = baseline
            self.return_normalized = True

    def __call__(self, alternative_fnr_fpr_pair: Tensor, baseline_fnr_fpr_pair: Tensor):

        norm_change = self.normalized_change(alternative_fnr_fpr_pair, baseline_fnr_fpr_pair)
        avg_fpr_fnr = self.average_values(norm_change)
        cev = self.squared_euclidean(norm_change, avg_fpr_fnr)

        return cev[0]

    @staticmethod
    def normalized_change(alternative_model: Tensor, baseline_model: Tensor) -> Tensor:
        return (alternative_model - baseline_model) / baseline_model

    @staticmethod
    def average_values(normalized_change: Tensor) -> Tensor:
        return torch.mean(normalized_change, dim=1, keepdim=True)

    @staticmethod
    def squared_euclidean(norm_change: Tensor, avg_fpr_fnr: Tensor) -> Tensor:
        dist = 0
        avg_fpr = avg_fpr_fnr[0]
        avg_fnr = avg_fpr_fnr[1]

        for (mean_pos, mean_neg) in norm_change.T:
            dist += torch.sqrt((avg_fpr - mean_pos) ** 2 + (avg_fnr - mean_neg) ** 2)

        dist = dist ** 2 / norm_change.shape[1]

        return dist


class SDE:

    def __init__(self):
        pass

    def __call__(self, alternative_fnr_fpr_pair: Tensor, baseline_fnr_fpr_pair: Tensor) -> Tensor:
        comp_FPR = alternative_fnr_fpr_pair[0, :]
        comp_FNR = alternative_fnr_fpr_pair[1, :]

        orig_FPR = baseline_fnr_fpr_pair[0, :]
        orig_FNR = baseline_fnr_fpr_pair[1, :]

        delta_FPR_i = CEV.normalized_change(comp_FPR, orig_FPR)
        delta_FNR_i = CEV.normalized_change(comp_FNR, orig_FNR)

        sde = self.mean_absolute_change(delta_FPR_i, delta_FNR_i)

        return sde

    @staticmethod
    def mean_absolute_change(delta_fpr_i: Tensor, delta_fnr_i: Tensor) -> Tensor:
        return torch.sum(torch.abs(delta_fnr_i - delta_fpr_i))


def cev(alternative_fnr_fpr_pair: Tensor, baseline_fnr_fpr_pair: Tensor = None):
    if not baseline_fnr_fpr_pair:
        baseline_fnr_fpr_pair = torch.tensor([[0.5 for _ in range(alternative_fnr_fpr_pair.shape[1])],
                                              [0.5 for _ in range(alternative_fnr_fpr_pair.shape[1])]])

    normalized_change = CEV.normalized_change(alternative_fnr_fpr_pair, baseline_fnr_fpr_pair)
    avg_fpr_fnr = CEV.average_values(normalized_change)
    cev_final = CEV.squared_euclidean(normalized_change, avg_fpr_fnr)

    return cev_final[0]


def sde(alternative_fnr_fpr_pair: Tensor, baseline_fnr_fpr_pair: Tensor = None) -> Tensor:
    """

    Args:
        alternative_fnr_fpr_pair: Alternative model
        baseline_fnr_fpr_pair: baseline fpr/fnr values. If None fpr/fnr pair is set to 0.5 for all alternative
        fpr/fnr pairs (Default: None)
    Returns:

    """

    if not baseline_fnr_fpr_pair:
        baseline_fnr_fpr_pair = torch.tensor([[0.5 for _ in range(alternative_fnr_fpr_pair.shape[1])],
                                              [0.5 for _ in range(alternative_fnr_fpr_pair.shape[1])]])

    comp_FPR = alternative_fnr_fpr_pair[0, :]
    comp_FNR = alternative_fnr_fpr_pair[1, :]

    orig_FPR = baseline_fnr_fpr_pair[0, :]
    orig_FNR = baseline_fnr_fpr_pair[1, :]

    delta_FPR_i = CEV.normalized_change(comp_FPR, orig_FPR)
    delta_FNR_i = CEV.normalized_change(comp_FNR, orig_FNR)

    sde_final = SDE.mean_absolute_change(delta_FPR_i, delta_FNR_i)

    return sde_final