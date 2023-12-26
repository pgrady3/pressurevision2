import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
from prediction.model.gradient_reversal_layer import GradientReverseLayerF


def get_dann_model():
    layers = []
    in_channels = 2048
    out_channels = 2048
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(Flatten())
    layers.append(MLP(2048, [512, 2], nn.BatchNorm1d, dropout=0.2, inplace=False))
    return nn.Sequential(*layers)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class BottleneckClassifierModel(nn.Module):
    def __init__(self, num_out_logits):
        super(BottleneckClassifierModel, self).__init__()
        in_channels = 2048
        out_channels = 2048

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = MLP(2048, [512, num_out_logits], nn.BatchNorm1d, dropout=0.2, inplace=False)

    def forward(self, feature_list):
        x = feature_list[-1]

        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)

        x = torch.sigmoid(x)

        return x