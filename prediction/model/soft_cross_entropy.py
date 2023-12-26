from torch.nn.modules.loss import _Loss
from torch import Tensor
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import CrossEntropyLoss

# Explained in these papers: https://arxiv.org/pdf/1609.03894.pdf
# https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Su_Render_for_CNN_ICCV_2015_paper.pdf
# Basically a soft version of cross entropy that penalizes near misses less


class SoftCrossEntropyLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, omega, num_classes, weight=None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super(SoftCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.omega = omega
        self.num_classes = num_classes

    def forward(self, input: Tensor, target_indexes: Tensor) -> Tensor:
        # target is a array of GT indexes
        # turn it into a full-size array of probabilities
        target = torch.arange(0, self.num_classes, dtype=input.dtype, device=input.device)

        # expand along batch dim
        target = target.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        target = target.expand(input.shape[0], -1, input.shape[2], input.shape[3])

        target = torch.abs(target - target_indexes.unsqueeze(1))   # Get distance from true label
        target = torch.exp(-target / self.omega)

        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)


if __name__ == "__main__":
    targets = torch.ones(32, 448, 448, dtype=torch.long)
    inputs = torch.rand(32, 9, 448, 448)

    loss = SoftCrossEntropyLoss(1, 9)
    loss2 = CrossEntropyLoss()
    output = loss(inputs, targets)
    output2 = loss2(inputs, targets)
    print(float(output), float(output2))