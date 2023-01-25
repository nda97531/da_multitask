import torch as tr
import torch.nn as nn


class CompleteModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: tr.Tensor):
        """

        Args:
            x: [batch, ..., channel]

        Returns:

        """
        # change to [batch, channel, ...]
        x = tr.permute(x, [0, 2, 1])

        x = self.backbone(x)
        x = self.classifier(x)
        return x

# class MultiTaskModel(nn.Module):
#     def __init__(self, backbone: nn.Module, classifiers: nn.ModuleList) -> None:
#         super().__init__()
#         self.backbone = backbone
#         self.classifiers = classifiers
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.classifier(x)
#         return x
