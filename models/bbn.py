from torch.distributions import normal
from torch.nn.parameter import Parameter
import torch
import math
import collections
from itertools import repeat
from torch import nn
import torch.nn.functional as F

def _pair(x):
    if isinstance(x, collections.Iterable):
        return x
    return tuple(repeat(x, 2))

class VarLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, std=0.05):
        super(VarLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mean = Parameter(torch.Tensor(out_features, in_features))
        self.weight_logsigma = Parameter(torch.Tensor(out_features, in_features))
        self.eps = normal.Normal(0., std)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_mean.size(1))
        self.weight_mean.data.uniform_(-stdv, stdv)
        self.weight_logsigma.data.normal_(0, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.training:
            weight_sigma = torch.log(1 + torch.exp(self.weight_logsigma))
            #mean = torch.nn.functional.linear(input, self.weight_mean, self.bias)
            #sigma = torch.sqrt(torch.nn.functional.linear(torch.pow(input, 2), torch.pow(weight_sigma, 2)))
            #eps = self.eps.sample(mean.size()).cuda()
            #output = mean + sigma * eps
            eps = self.eps.sample(weight_sigma.size()).cuda()
            weight = self.weight_mean + weight_sigma * eps
            output = torch.nn.functional.linear(input, weight, self.bias)
            return output
        else:
            output = torch.nn.functional.linear(input, self.weight_mean, self.bias)
            return output

    def get_KL(self):
        weight_sigma = torch.log(1 + torch.exp(self.weight_logsigma))
        KL_elems = - 0.5 - torch.log(weight_sigma) + 0.5 * self.weight_mean ** 2 + 0.5 * weight_sigma ** 2  
        return torch.sum(KL_elems)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class VarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, std=0.05):
        super(VarConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_padding = _pair(0)
        self.groups = groups
        self.weight_mean = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.weight_logsigma = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.eps = normal.Normal(0., std)

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight_mean.data.uniform_(-stdv, stdv)
        self.weight_logsigma.data.normal_(0, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def forward(self, input):
        if self.training:
            weight_sigma = torch.log(1 + torch.exp(self.weight_logsigma))
            eps = self.eps.sample(weight_sigma.size()).cuda()
            weight = self.weight_mean + weight_sigma * eps
            return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight_mean, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def get_KL(self):
        weight_sigma = torch.log(1 + torch.exp(self.weight_logsigma))
        KL_elems = - 0.5 - torch.log(weight_sigma) + 0.5 * self.weight_mean ** 2 + 0.5 * weight_sigma ** 2  
        return torch.sum(KL_elems)