from typing import Union
import torch.nn as nn


class Conv1dBlock(nn.Module):

    def __init__(self, in_filters: int, out_filters: int,
                 kernel_size: int, stride: int = 1,
                 padding: Union[int, list, tuple] = 0, dilation: int = 1,
                 drop_rate: float = 0., use_spatial_dropout: bool = False,
                 activation=nn.ReLU(), norm_layer: Union[str, None] = 'batch', input_len: Union[int, None] = None):
        """
        Conv with padding, activation, batch norm

        Args:
            input_len: length of the input window, this is only necessary if `norm_layer` is 'layer'
            in_filters:
            out_filters:
            kernel_size:
            stride:
            padding:
            dilation:
            drop_rate:
            use_spatial_dropout:
            activation:
            norm_layer:
        """
        super().__init__()
        conv = []

        if type(padding) is int:
            conv.append(nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size,
                                  stride=stride, padding=padding, dilation=dilation))

        elif (type(padding) is tuple) or (type(padding) is list):
            conv += [nn.ConstantPad1d(padding=padding, value=0.),
                     nn.Conv1d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size,
                               stride=stride, dilation=dilation)]
        else:
            raise ValueError('padding must be integer or list/tuple!')

        conv.append(activation)

        if norm_layer == 'batch':
            conv.append(nn.BatchNorm1d(out_filters))
        elif norm_layer == 'layer':
            assert input_len > 0
            conv.append(nn.LayerNorm([out_filters, input_len]))
        elif norm_layer:
            raise ValueError(f'invalid norm_layer: {norm_layer}')

        if use_spatial_dropout:
            conv.append(nn.Dropout1d(drop_rate))
        else:
            conv.append(nn.Dropout(drop_rate))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        x = self.conv(x)
        return x

# class Conv2dBlock(nn.Module):
#
#     def __init__(self, in_filters, out_filters, kernel_size, stride=1, padding=0, dilation=1, drop_rate=0.,
#                  activation=nn.ReLU(), use_batchnorm=True):
#         super().__init__()
#         """
#         Conv with padding, activation, batch norm
#         """
#         conv = []
#
#         if type(padding) is int:
#             conv += [nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size,
#                                stride=stride, padding=padding, dilation=dilation), ]
#
#         elif (type(padding) is tuple) or (type(padding) is list):
#             conv += [nn.ConstantPad2d(padding=padding, value=0.),
#                      nn.Conv2d(in_channels=in_filters, out_channels=out_filters, kernel_size=kernel_size,
#                                stride=stride, dilation=dilation)]
#         else:
#             raise ValueError('padding must be integer or list/tuple!')
#
#         if use_batchnorm:
#             conv += [activation,
#                      nn.BatchNorm2d(out_filters),
#                      nn.Dropout(drop_rate)]
#         else:
#             conv += [activation,
#                      nn.Dropout(drop_rate)]
#
#         self.conv = nn.Sequential(*conv)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
