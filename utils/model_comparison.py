import numpy as np
import torch
from torch import Tensor


class CEV:

    def __init__(self, baseline: Tensor = None):
        if baseline:
            self.baseline_model_pair = baseline
            self.return_normalized = True

    def __call__(self, alternative_fnr_fpr_pair: Tensor, baseline_fnr_fpr_pair: Tensor):

        norm_change = self.normalized_change(alternative_fnr_fpr_pair, baseline_fnr_fpr_pair)
        print(norm_change)
        avg_fpr_fnr = self.average_values(norm_change)
        cev = self.squared_euclidean(norm_change, avg_fpr_fnr)

        return cev[0]

    @staticmethod
    def normalized_change(alternative_model: Tensor, baseline_model: Tensor) -> Tensor:
        return (alternative_model - baseline_model) / baseline_model

    @staticmethod
    def average_values(normalized_change: Tensor) -> Tensor:
        return torch.sum(normalized_change, dim=1, keepdim=True)

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


def cev(alternative_fnr_fpr_pair: Tensor, baseline_fnr_fpr_pair: Tensor):
    normalized_change = CEV.normalized_change(alternative_fnr_fpr_pair, baseline_fnr_fpr_pair)
    avg_fpr_fnr = CEV.average_values(normalized_change)
    cev_final = CEV.squared_euclidean(normalized_change, avg_fpr_fnr)

    return cev_final[0]


def sde(alternative_fnr_fpr_pair: Tensor, baseline_fnr_fpr_pair: Tensor) -> Tensor:
    """

    Args:
        alternative_fnr_fpr_pair:
        baseline_fnr_fpr_pair:

    Returns:

    """
    comp_FPR = alternative_fnr_fpr_pair[0, :]
    comp_FNR = alternative_fnr_fpr_pair[1, :]

    orig_FPR = baseline_fnr_fpr_pair[0, :]
    orig_FNR = baseline_fnr_fpr_pair[1, :]

    delta_FPR_i = CEV.normalized_change(comp_FPR, orig_FPR)
    delta_FNR_i = CEV.normalized_change(comp_FNR, orig_FNR)

    sde_final = SDE.mean_absolute_change(delta_FPR_i, delta_FNR_i)

    return sde_final


if __name__ == '__main__':
    com_model_pair = torch.tensor([[.08, .01, .03], [.07, .04, .01]])  # FPR / FNR for each class new model
    ori_mode_pair = torch.tensor([[.11, .09, .08], [.1, .09, .06]])  # FPR / FNR for each class old model
    print(com_model_pair[0, :])
    cev = CEV()
    sde = SDE()

    print(cev(com_model_pair, ori_mode_pair))
    print(sde(com_model_pair, ori_mode_pair))
