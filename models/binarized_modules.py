import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np


def Binarize(x, quant_mode='det'):
    """
    二值化
    det模式，直接返回x的1,-1,0

    其他模式，((x+1)/2 + (rand(size)-0.5)).clamp(0,1).round() * 2 - 1
    
    _: 就地操作，直接改变原变量
    clamp: 限制变量为0,1，最小值限定为0，最大值限定为1
    round: 四舍五入
    """
    if quant_mode == 'det':
        return torch.sign(x)
    else:
        return x.add_(1).div_(2).add_(torch.rand(x.size()).add(-0.5)).clamp_(
            0, 1).round().mul_(2).add_(-1)


class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.margin = 1.0

    def hinge_loss(self, input_, target):
        #import pdb; pdb.set_trace()
        output = self.margin - input_.mul(target)
        output[output.le(0)] = 0
        return output.mean()

    def forward(self, input_, target):
        return self.hinge_loss(input_, target)


class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction, self).__init__()
        self.margin = 1.0

    def forward(self, input_, target):
        output = self.margin - input_.mul(target)
        output[output.le(0)] = 0
        self.save_for_backward(input_, target)
        loss = output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self, grad_output):
        input_, target = self.saved_tensors
        output = self.margin - input_.mul(target)
        output[output.le(0)] = 0
        import pdb
        pdb.set_trace()
        grad_output.resize_as_(input_).copy_(target).mul_(-2).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(input_.numel())
        return grad_output, grad_output


def Quantize(tensor, quant_mode='det', params=None, numBits=8):
    tensor.clamp_(-2**(numBits - 1), 2**(numBits - 1))
    if quant_mode == 'det':
        tensor = tensor.mul(2**(numBits - 1)).round().div(2**(numBits - 1))
    else:
        tensor = tensor.mul(2**(numBits - 1)).round().add(
            torch.rand(tensor.size()).add(-0.5)).div(2**(numBits - 1))
        quant_fixed(tensor, params)
    return tensor


class BinarizeLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input_):

        if input_.size(1) != 784:
            input_.data = Binarize(input_.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)
        out = F.linear(input_, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BinarizeConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input_):
        if input_.size(1) != 3:
            input_.data = Binarize(input_.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        out = F.conv2d(input_, self.weight, None, self.stride, self.padding,
                       self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
