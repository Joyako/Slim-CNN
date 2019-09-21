import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )


class DWSeparableConv(nn.Module):
    """Depth-Wise Separable Convolution"""
    def __init__(self, inp, oup):
        super().__init__()
        self.dwc = ConvBNReLU(inp, inp, kernel_size=3, groups=inp)
        self.pwc = ConvBNReLU(inp, oup, kernel_size=1)

    def forward(self, x):
        x = self.dwc(x)
        x = self.pwc(x)

        return x


class SSEBlock(nn.Module):
    """Separable Squeeze-Expand Block"""
    def __init__(self, inp, oup):
        super().__init__()
        out_channel = oup * 4
        self.pwc1 = ConvBNReLU(inp, oup, kernel_size=1)
        self.pwc2 = ConvBNReLU(oup, out_channel, kernel_size=1)
        self.dwc = DWSeparableConv(oup, out_channel)

    def forward(self, x):
        x = self.pwc1(x)
        out1 = self.pwc2(x)
        out2 = self.dwc(x)

        return torch.cat((out1, out2), 1)


class SlimModule(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()
        hidden_dim = oup * 4
        out_channel = oup * 3
        self.sse1 = SSEBlock(inp, oup)
        self.sse2 = SSEBlock(hidden_dim * 2, oup)
        self.dwc = DWSeparableConv(hidden_dim * 2, out_channel)
        self.conv = ConvBNReLU(inp, hidden_dim * 2, kernel_size=1)

    def forward(self, x):
        out = self.sse1(x)
        out += self.conv(x)
        out = self.sse2(out)
        out = self.dwc(out)

        return out


class SlimNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = ConvBNReLU(3, 96, kernel_size=7, stride=2)
        self.max_pool0 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 4 stacked Slim Module
        self.module1 = SlimModule(96, 16)
        self.module2 = SlimModule(48, 32)
        self.module3 = SlimModule(96, 48)
        self.module4 = SlimModule(144, 64)

        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x):
        x = self.max_pool0(self.conv(x))
        x = self.max_pool1(self.module1(x))
        x = self.max_pool2(self.module2(x))
        x = self.max_pool3(self.module3(x))
        x = self.max_pool4(self.module4(x))

        # global average pooling
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.sigmoid(x)

        return x

from torchstat import stat
net =SlimNet(4)
stat(net, (3, 178, 218))






