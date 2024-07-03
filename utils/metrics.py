import numpy as np
from math import sqrt
from scipy import stats
import torch.nn as nn
import torch


class MAPELoss(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_pred - y_true) / y_true)) * 100


class RSELoss(nn.Module):
    def __int__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2
        squared_sum_observed = torch.sum((y_true - torch.mean(y_true)) ** 2)
        rse = torch.sum(squared_error) / squared_sum_observed
        return rse


def get_rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def get_mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def get_pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def get_spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def get_cindex(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)

    return CI


def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / float(sum(y_pred * y_pred))


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)


def get_rmspe(y_true, y_pred):
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


if __name__ == '__main__':
    y_true = np.array([1, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 1])

    rmse = get_rmse(y_true, y_pred)
    mse = get_mse(y_true, y_pred)
    pearson = get_pearson(y_true, y_pred)
    spearman = get_spearman(y_true, y_pred)
    r2 = r_squared_error(y_true, y_pred)
    rm2 = get_rm2(y_true, y_pred)
    CI = get_cindex(y_true, y_pred)

    print(rmse)
    print(mse)
    print(pearson)
    print(spearman)
    print(r2)
    print(rm2)
    print(CI)

# %%
