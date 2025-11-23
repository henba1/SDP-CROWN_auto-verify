import torch
import torch.nn as nn
import torch.nn.functional as F

"""
JAIR" refers to CIFAR-10 and MNIST models used in:
https://jair.org/index.php/jair/article/view/18403.
Model weights can be found in the accompanying research repo:
https://github.com/ADA-research/NNV_JAIR_robustness_distributions/tree/main.
"""

def MNIST_MLP():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    return model


class MNIST_ConvSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x_p = torch.relu(self.fc1(x))
        x = self.fc2(x_p)
        return x


class MNIST_ConvLarge(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MNIST_NN(nn.Module):
    """JAIR 3-layer fully-connected MNIST network."""

    def __init__(self):
        super().__init__()
        self.name = "mnist_nn"
        self.layer1 = nn.Linear(784, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class MNIST_RELU_4_1024(nn.Module):
    """JAIR 4-layer fully-connected MNIST network with 1024 ReLU units per hidden layer."""

    def __init__(self):
        super().__init__()
        self.name = "mnist_relu_4_1024"
        self.layer1 = nn.Linear(784, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layer4 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x


def CIFAR10_CNN_A():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    return model


def CIFAR10_CNN_B():
    return nn.Sequential(
        nn.ZeroPad2d((1, 2, 1, 2)),
        nn.Conv2d(3, 32, (5, 5), stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 128, (4, 4), stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8192, 250),
        nn.ReLU(),
        nn.Linear(250, 10),
    )


def CIFAR10_CNN_C():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(8, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(576, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    return model


def CIFAR10_ConvSmall():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, stride=2, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(32 * 6 * 6, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    return model


def CIFAR10_ConvDeep():
    model = nn.Sequential(
        nn.Conv2d(3, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * 8 * 8, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )
    return model


class CIFAR10_ConvLarge(nn.Module):
    """SDP-CROWN 'large' CIFAR-10 conv net, architecturally matching JAIR's CONV_BIG."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# JAIR CIFAR-10 architectures replicated here so their checkpoints can be
# loaded directly by SDP-CROWN.
class CONV_BIG(nn.Module):
    """JAIR ConvBig architecture for CIFAR-10."""

    def __init__(self):
        super().__init__()
        self.name = "conv_big"
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(32, 32, 4, padding=1, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(64, 64, 4, padding=1, stride=2)
        self.fc1 = nn.Linear(4096, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CIFAR_7_1024(nn.Module):
    """JAIR 7-layer fully-connected CIFAR-10 network."""

    def __init__(self):
        super().__init__()
        self.name = "cifar_7_1024"
        self.fc1 = nn.Linear(3072, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fc7 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(-1, 3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        return x


class BasicBlock(nn.Module):
    """Basic residual block as used in JAIR ResNet-4B (from VNN-COMP 2021 ResNet)."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, kernel=3):
        super().__init__()
        self.bn = bn
        if kernel == 3:
            self.conv1 = nn.Conv2d(
                in_planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=not self.bn,
            )
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.bn,
            )
        elif kernel == 2:
            self.conv1 = nn.Conv2d(
                in_planes,
                planes,
                kernel_size=2,
                stride=stride,
                padding=1,
                bias=not self.bn,
            )
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=2,
                stride=1,
                padding=0,
                bias=not self.bn,
            )
        elif kernel == 1:
            self.conv1 = nn.Conv2d(
                in_planes,
                planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=not self.bn,
            )
            if self.bn:
                self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(
                planes,
                planes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=not self.bn,
            )
        else:
            raise ValueError("Unsupported kernel size for BasicBlock")

        if self.bn:
            self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if self.bn:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=not self.bn,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=not self.bn,
                    )
                )

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet5(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=10, in_planes=64, bn=True, last_layer="avg"):
        super().__init__()
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=2, padding=1, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes * 2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 8 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 8 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            raise ValueError("last_layer type not supported for ResNet5")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_val in strides:
            layers.append(block(self.in_planes, planes, stride_val, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.last_layer == "dense":
            out = torch.flatten(out, 1)
            out = F.relu(self.linear1(out))
            out = self.linear2(out)
        return out


class ResNet9(nn.Module):
    
    def __init__(
        self,
        block,
        num_blocks=2,
        num_classes=10,
        in_planes=64,
        bn=True,
        last_layer="avg",
        name="resnet_9",
    ):
        super().__init__()
        self.name = name
        self.in_planes = in_planes
        self.bn = bn
        self.last_layer = last_layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=2, padding=1, bias=not self.bn)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes * 2, num_blocks, stride=2, bn=bn, kernel=3)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks, stride=2, bn=bn, kernel=3)
        if self.last_layer == "avg":
            self.avg2d = nn.AvgPool2d(4)
            self.linear = nn.Linear(in_planes * 2 * block.expansion, num_classes)
        elif self.last_layer == "dense":
            self.linear1 = nn.Linear(in_planes * 2 * block.expansion * 16, 100)
            self.linear2 = nn.Linear(100, num_classes)
        else:
            raise ValueError("last_layer type not supported for ResNet9")

    def _make_layer(self, block, planes, num_blocks, stride, bn, kernel):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_val in strides:
            layers.append(block(self.in_planes, planes, stride_val, bn, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.bn:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        if self.last_layer == "avg":
            out = self.avg2d(out)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        elif self.last_layer == "dense":
            out = torch.flatten(out, 1)
            out = F.relu(self.linear1(out))
            out = self.linear2(out)
        return out


def ResNet2B():
    return ResNet5(BasicBlock, num_blocks=2, in_planes=8, bn=False, last_layer="dense", name="resnet_2b")


def ResNet4B(bn: bool = False):
    """JAIR ResNet-4B wrapper, matching the training-time constructor."""

    return ResNet9(BasicBlock, num_blocks=2, in_planes=16, bn=bn, last_layer="dense", name="resnet_4b")
