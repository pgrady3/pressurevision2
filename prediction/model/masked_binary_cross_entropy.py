from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch
import torch.nn.functional as F


class MaskedBCELoss(_Loss):
    # Copy from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#BCELoss
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(MaskedBCELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        mask = torch.ones_like(target)
        mask[target < 0] = 0
        # print('mask, target', mask, target)

        return F.binary_cross_entropy(input, target, weight=mask, reduction=self.reduction)