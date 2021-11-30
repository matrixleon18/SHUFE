# encoding=GBK


import torch


class Mul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, x, b, x_requies_grad=True):
        ctx.x_requires_grad = x_requies_grad
        ctx.save_for_backward(w, x)
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_variables
        grad_w = grad_output * x
        if ctx.x_requires_grad:
            grad_x = grad_output * x
        else:
            grad_x = None
        grad_b = grad_output * 1
        return grad_w, grad_x, grad_b, None