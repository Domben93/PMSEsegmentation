import numpy as np
from numpy import ndarray


class CEV:

    def __init__(self, baseline: ndarray = None):
        if baseline:
            self.baseline_model_pair = baseline
            self.return_normalized = True

    def __call__(self, comparison_model_pair: ndarray, original_model_pair: ndarray):
        norm_change = self.normalized_change(comparison_model_pair, original_model_pair)
        avg_fpr_fnr = self.average_values(norm_change)
        cev = self.squared_euclidean(norm_change, avg_fpr_fnr)

        return cev[0]

    @staticmethod
    def normalized_change(comp_model_pair: ndarray, orig_model_pair: ndarray) -> ndarray:
        return (comp_model_pair - orig_model_pair) / orig_model_pair

    @staticmethod
    def average_values(normalized_change: ndarray) -> ndarray:
        return np.sum(normalized_change, axis=1, keepdims=True)

    @staticmethod
    def squared_euclidean(norm_change: ndarray, avg_fpr_fnr: ndarray) -> ndarray:
        dist = 0
        avg_fpr = avg_fpr_fnr[0]
        avg_fnr = avg_fpr_fnr[1]

        for (mean_pos, mean_neg) in norm_change.T:
            dist += (avg_fpr - mean_pos)**2 + (avg_fnr - mean_neg)**2

        dist = dist**2 / norm_change.shape[1]

        return dist


class SDE:

    def __init__(self, baseline: ndarray = None):
        self.baseline_model = baseline

    def __call__(self, comparison_model_pair: ndarray, original_model_pair: ndarray) -> float:
        comp_FPR = comparison_model_pair[0, :]
        comp_FNR = comparison_model_pair[1, :]

        orig_FPR = original_model_pair[0, :]
        orig_FNR = original_model_pair[1, :]

        delta_FPR_i = CEV.normalized_change(comp_FPR, orig_FPR)
        delta_FNR_i = CEV.normalized_change(comp_FNR, orig_FNR)

        sde = self.mean_absolute_change(delta_FPR_i, delta_FNR_i)
        print(sde)
        return sde

    @staticmethod
    def mean_absolute_change(delta_fpr_i, delta_fnr_i):
        return np.sum(np.abs(delta_fnr_i - delta_fpr_i))


if __name__ == '__main__':

    com_model_pair = np.array([[.08, .01, .03], [.07, .04, .01]])  # FPR / FNR for each class new model
    ori_mode_pair = np.array([[.11, .09, .08], [.1, .09, .06]])  # FPR / FNR for each class old model
    print(com_model_pair[0, :])
    cev = CEV()
    sde = SDE()

    cev(com_model_pair, ori_mode_pair)
    sde(com_model_pair, ori_mode_pair)
