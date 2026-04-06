import torch
import torch.nn.functional as F


def normalize_features_l2(x):
    return F.normalize(x, p=2, dim=-1)
