import torch
import torch.nn as nn

from enum import Enum

from apec import APEC, MAPEC


class CIFAR100Activations(Enum):
    APEC = APEC
    MAPEC = MAPEC
    MISH = nn.Mish
    SOFTPLUS = nn.Softplus
    SELU = nn.SELU
    GELU = nn.GELU
    TANH = nn.Tanh
    RELU = nn.ReLU
    PRELU = nn.PReLU
    LEAKYRELU = nn.LeakyReLU
    RELU6 = nn.ReLU6
    ELU = nn.ELU
    SIGMOID = nn.Sigmoid
    TANHSHRINK = nn.Tanhshrink
    HARDTANH = nn.Hardtanh
    HARDSIGMOID = nn.Hardsigmoid
    HARDSWISH = nn.Hardswish


class CIFAR100(nn.Module):
    def __init__(
        self,
        activation_fn: CIFAR100Activations = CIFAR100Activations.APEC,
    ):
        super(CIFAR100, self).__init__()

        print(f"Model's activation function: {activation_fn.name}")

        activation_fn = activation_fn.value

        self.a_conv_pre = nn.Conv2d(3, 32, 3, 1, 1)
        self.a_activation_pre = activation_fn()
        self.a_conv_post = nn.Conv2d(32, 32, 3, 1, 1)
        self.a_activation_post = activation_fn()
        self.a_pooling = nn.MaxPool2d(2, 2)
        self.a_layer_norm = nn.LayerNorm([16, 16])

        self.b_conv_pre = nn.Conv2d(32, 64, 3, 1, 1)
        self.b_activation_pre = activation_fn()
        self.b_conv_post = nn.Conv2d(64, 64, 3, 1, 1)
        self.b_activation_post = activation_fn()
        self.b_pooling = nn.MaxPool2d(2, 2)
        self.b_layer_norm = nn.LayerNorm([8, 8])

        self.c_conv_pre = nn.Conv2d(64, 128, 3, 1, 1)
        self.c_activation_pre = activation_fn()
        self.c_conv_post = nn.Conv2d(128, 128, 3, 1, 1)
        self.c_activation_post = activation_fn()
        self.c_pooling = nn.MaxPool2d(2, 2)
        self.c_layer_norm = nn.LayerNorm([4, 4])

        self.reductor_fc = nn.Linear(128, 32)
        self.reductor_activation = activation_fn()
        self.reductor_layer_norm = nn.LayerNorm([4, 4])

        self.d_fc = nn.Linear(512, 256)
        self.d_activation = activation_fn()
        self.d_batch_norm = nn.BatchNorm1d(256)

        self.e_fc = nn.Linear(256, 100)
        self.e_activation = activation_fn()
        self.e_batch_norm = nn.BatchNorm1d(100)

        self.f_fc = nn.Linear(100, 100)

        self.dropout = nn.Dropout(p=0.5)
        self.log_softmax = nn.LogSoftmax(dim=1)

        pass

    def forward(
        self,
        x: torch.Tensor,
    ):
        x = self.a_conv_pre(x)
        x_pre = self.a_activation_pre(x)
        x = self.a_conv_post(x_pre)
        x = self.a_activation_post(x)
        x = x_pre + x
        x = self.a_pooling(x)
        x = self.a_layer_norm(x)

        x = self.b_conv_pre(x)
        x_pre = self.b_activation_pre(x)
        x = self.b_conv_post(x_pre)
        x = self.b_activation_post(x)
        x = x_pre + x
        x = self.b_pooling(x)
        x = self.b_layer_norm(x)

        x = self.c_conv_pre(x)
        x_pre = self.c_activation_pre(x)
        x = self.c_conv_post(x_pre)
        x = self.c_activation_post(x)
        x = x_pre + x
        x = self.c_pooling(x)
        x = self.c_layer_norm(x)

        x = self.reductor_fc(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.reductor_activation(x)
        x = self.reductor_layer_norm(x)

        x = self.dropout(x)

        x = x.flatten(1)
        x = self.d_fc(x)
        x = self.d_activation(x)
        x = self.d_batch_norm(x)

        x = self.e_fc(x)
        x = self.e_activation(x)
        x = self.e_batch_norm(x)

        x = self.f_fc(x)

        x = self.log_softmax(x)

        return x
