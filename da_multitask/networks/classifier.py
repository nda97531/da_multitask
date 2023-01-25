import torch.nn as nn


class FCClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        """

        Args:
            n_features:
            n_classes:
        """
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class MultiFCClassifiers(nn.Module):
    def __init__(self, n_features: int, n_classes: list):
        """

        Args:
            n_features:
            n_classes:
        """
        super().__init__()
        self.fcs = nn.ModuleList()
        for n_c in n_classes:
            self.fcs.append(nn.Linear(n_features, n_c))

    def forward(self, x):
        output = [fc(x) for fc in self.fcs]
        return output
