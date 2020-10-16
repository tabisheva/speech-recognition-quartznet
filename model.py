import torch.nn as nn


class QuartzNetBlock(nn.Module):
    def __init__(self, feat_in, filters, repeat=3, kernel_size=11, stride=1,
                 dilation=1, residual=True, separable=False):
        super(QuartzNetBlock, self).__init__()
        self.res = nn.Sequential(nn.Conv1d(feat_in, filters, kernel_size=1),
                                     nn.BatchNorm1d(filters)) if residual else None
        self.conv = nn.ModuleList()
        for idx in range(repeat):
            self.conv.extend(
                self._get_conv_bn_layer(
                    feat_in,
                    filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    separable=separable))
            if (idx != repeat - 1 and residual):
                self.conv.extend([nn.ReLU(), nn.Dropout(p=0.2)])
            feat_in = filters
        self.out = nn.Sequential(nn.ReLU(), nn.Dropout(p=0.2))

    def _get_conv_bn_layer(self, in_channels, out_channels, kernel_size,
                           stride=1, dilation=1, separable=False):
        if dilation > 1:
            same_padding = (dilation * kernel_size) // 2 - 1
        else:
            same_padding = kernel_size // 2
        if separable:
            layers = [
                nn.Conv1d(in_channels, in_channels, kernel_size,
                             groups=in_channels, stride=stride, dilation=dilation, padding=same_padding),
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
            ]
        else:
            layers = [
                nn.Conv1d(in_channels, out_channels, kernel_size,
                             stride=stride, dilation=dilation, padding=same_padding)
            ]
        layers.append(nn.BatchNorm1d(out_channels))
        return layers

    def forward(self, inputs):
        inputs_for_res = inputs
        for layer in self.conv:
            inputs = layer(inputs)
        if self.res is not None:
            inputs = inputs + self.res(inputs_for_res)
        inputs = self.out(inputs)
        return inputs

class QuartzNet(nn.Module):
    def __init__(self, quartznet_conf, feat_in, num_classes):
        super(QuartzNet, self).__init__()
        layers = []
        for block_conf in quartznet_conf:
            layers.append(
                QuartzNetBlock(feat_in,
                               block_conf['filters'],
                               repeat=block_conf['repeat'],
                               kernel_size=block_conf['kernel'],
                               stride=block_conf['stride'],
                               dilation=block_conf['dilation'],
                               residual=block_conf['residual'],
                               separable=block_conf['separable']))
            feat_in = block_conf['filters']
        layers.append(nn.Conv1d(feat_in, num_classes, kernel_size=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)
