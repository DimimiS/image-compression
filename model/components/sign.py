from torch.autograd import Function
import torch.nn as nn


class Sign_Function(Function):
    def __init__(self):
        super(Sign_Function, self).__init__()

    @staticmethod
    def forward(ctx, input, is_training=True):
        # Apply quantization noise while only training
        if is_training:
            prob = input.new(input.size()).uniform_()
            x = input.clone()
            x[(1 - input) / 2 <= prob] = 1
            x[(1 - input) / 2 > prob] = -1
            return x
        else:
            return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
    

class Sign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return Sign_Function.apply(x, self.training)
