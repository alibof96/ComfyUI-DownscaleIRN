import torch
import torch.nn as nn

def _init_conv(m: nn.Module, scale: float = 1.0, method: str = "kaiming"):
    if isinstance(m, nn.Conv2d):
        if method == "xavier":
            nn.init.xavier_normal_(m.weight, gain=1.0)
        else:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        m.weight.data *= scale
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

def initialize_weights(net, scale=1.0):
    if isinstance(net, (list, tuple)):
        for m in net:
            _init_conv(m, scale=scale, method="kaiming")
    else:
        net.apply(lambda m: _init_conv(m, scale=scale, method="kaiming"))

def initialize_weights_xavier(net, scale=1.0):
    if isinstance(net, (list, tuple)):
        for m in net:
            _init_conv(m, scale=scale, method="xavier")
    else:
        net.apply(lambda m: _init_conv(m, scale=scale, method="xavier"))