import torch.nn as nn


class Conv1dBlock(nn.Module):

    def __init__(self, in_filters, out_filters, kernel_size, stride=1, padding=0, dilation=1,
                 drop_rate=0., use_spatial_dropout=False,
                 activation=nn.ReLU(), use_batchnorm=True):
        super().__init__()
        """
        Conv with padding, activation, batch norm
        """
        conv = []

        if type(padding) is int:
            conv += [nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation), ]

        elif (type(padding) is tuple) or (type(padding) is list):
            conv += [nn.ConstantPad1d(padding=padding, value=0.),
                     nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size,
                               stride=stride, dilation=dilation)]
        else:
            raise ValueError('padding must be integer or list/tuple!')

        if use_batchnorm:
            conv += [activation,
                     nn.BatchNorm1d(out_filters)]
        else:
            conv.append(activation)

        if use_spatial_dropout:
            conv.append(nn.Dropout2d(drop_rate))
        else:
            conv.append(nn.Dropout(drop_rate))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv2dBlock(nn.Module):

    def __init__(self, in_filters, out_filters, kernel_size, stride=1, padding=0, dilation=1, drop_rate=0.,
                 activation=nn.ReLU(), use_batchnorm=True):
        super().__init__()
        """
        Conv with padding, activation, batch norm
        """
        conv = []

        if type(padding) is int:
            conv += [nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation), ]

        elif (type(padding) is tuple) or (type(padding) is list):
            conv += [nn.ConstantPad2d(padding=padding, value=0.),
                     nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size,
                               stride=stride, dilation=dilation)]
        else:
            raise ValueError('padding must be integer or list/tuple!')

        if use_batchnorm:
            conv += [activation,
                     nn.BatchNorm2d(out_filters),
                     nn.Dropout(drop_rate)]
        else:
            conv += [activation,
                     nn.Dropout(drop_rate)]

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        x = self.conv(x)
        return x
