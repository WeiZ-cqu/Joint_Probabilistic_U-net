import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, stride=1, padding=1):
        super(Bottleneck, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out + residual
        return self.relu(out)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, training=None, mid_channels=None,
                       kernel_size=3, stride=1, padding=1, post=True):
        super(ResBlock, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(in_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )
        if post:
            self.out_conv.add_module('batchnorm', nn.BatchNorm2d(out_channels))
            self.out_conv.add_module('relu', nn.ReLU(inplace=True))
            
        self.conv.apply(init_weights)
        self.out_conv.apply(init_weights)
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = out + residual
        out = self.relu(out)
        return self.out_conv(out)

class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, drop_rate, training, add_dropout):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        if add_dropout:
            self.add_module('dropout1', nn.Dropout2d(p=0.5)),
        self.add_module('conv2', nn.Conv2d(num_input_features, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.training = training

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

class DenseBlock(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, training=True, num_layers=2,
                       growth_rate=32, drop_rate=0, add_dropout=False):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features + i * growth_rate, growth_rate, drop_rate,
                                training, add_dropout)
            self.add_module('denselayer%d' % (i + 1), layer)
            
        self.add_module('norm', nn.BatchNorm2d(num_input_features + (i+1) * growth_rate))
        self.add_module('relu', nn.ReLU(inplace=True))
        if add_dropout:
            self.add_module('dropout', nn.Dropout2d(p=0.5))
        self.add_module('conv', nn.Conv2d(num_input_features + (i+1) * growth_rate, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        

class SingleDenseBlock(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, training=True, kernel_size=3,
                       padding=1):
        super(SingleDenseBlock, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=kernel_size, stride=1, padding=padding, bias=False))

class TransitionDown(nn.Sequential):
    def __init__(self):
        super(TransitionDown, self).__init__()

        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class TransitionUp(nn.Sequential):
    def __init__(self):
        super(TransitionUp, self).__init__()
        
        self.add_module('up', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

class PlanarTransform(nn.Module):
    def __init__(self, dim, init_sigma=0.01):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim).normal_(0, init_sigma))
        self.w = nn.Parameter(torch.randn(1, dim).normal_(0, init_sigma))
        self.b = nn.Parameter(torch.randn(1).fill_(0))

    def forward(self, x, normalize_u=True):
        # allow for a single forward pass over all the transforms in the flows with a Sequential container
        if isinstance(x, tuple):
            z, sum_log_abs_det_jacobians = x
        else:
            z, sum_log_abs_det_jacobians = x, 0

        # normalize u s.t. w @ u >= -1; sufficient condition for invertibility
        u_hat = self.u
        if normalize_u:
            wtu = (self.w @ self.u.t()).squeeze()
            m_wtu = - 1 + torch.log1p(wtu.exp())
            u_hat = self.u + (m_wtu - wtu) * self.w / (self.w @ self.w.t())

        # compute transform
        f_z = z + u_hat * torch.tanh(z @ self.w.t() + self.b)
        # compute log_abs_det_jacobian
        psi = (1 - torch.tanh(z @ self.w.t() + self.b)**2) @ self.w
        det = 1 + psi @ u_hat.t()
        log_abs_det_jacobian = torch.log(torch.abs(det) + 1e-6).squeeze()
        sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian

        return f_z, sum_log_abs_det_jacobians

class CouplingLayer(nn.Module):
    """
    Implementation of the additive coupling layer from section 3.2 of the NICE
    paper.
    """
    def __init__(self, data_dim, hidden_dim, mask, num_layers=4):
        super().__init__()

        assert data_dim % 2 == 0

        self.mask = mask

        modules = [nn.Linear(data_dim, hidden_dim), nn.LeakyReLU(0.2)]
        for i in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Linear(hidden_dim, data_dim))

        self.m = nn.Sequential(*modules)

    def forward(self, x, logdet, invert=False):
        if not invert:
            x1, x2 = self.mask * x, (1. - self.mask) * x
            y1, y2 = x1, x2 + (self.m(x1) * (1. - self.mask))
            return y1 + y2, logdet

        # Inverse additive coupling layer
        y1, y2 = self.mask * x, (1. - self.mask) * x
        x1, x2 = y1, y2 - (self.m(y1) * (1. - self.mask))
        return x1 + x2, logdet


class ScalingLayer(nn.Module):
    """
    Implementation of the scaling layer from section 3.3 of the NICE paper.
    """
    def __init__(self, data_dim):
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.randn(1, data_dim, requires_grad=True))

    def forward(self, x, logdet, invert=False):
        log_det_jacobian = torch.sum(self.log_scale_vector)

        if invert:
            return torch.exp(- self.log_scale_vector) * x, logdet - log_det_jacobian

        return torch.exp(self.log_scale_vector) * x, logdet + log_det_jacobian
        
        
        
        
        
        
        
        
        
        
        

