import torch
import torch.nn as nn
from training.utils import calc_sparsity_loss

class iNPUGradientHack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, output, sign, magnitude_log):
        ctx.save_for_backward(sign, magnitude_log, output)

        return output
    
    @staticmethod
    def backward(ctx, grad_outputs):
        sign, magnitude_log, output = ctx.saved_tensors

        numel = output.numel()
        target = output - numel / 2 * grad_outputs

        target_log = torch.log(torch.clamp(torch.abs(target), min=1e-36))

        grad_magnitude_log = 2 / numel * (magnitude_log - target_log)
        return None, None, grad_magnitude_log


class iNPU(nn.Module):
    def __init__(self, in_dim, out_dim, device=None, epsilon=1e-36):
        super().__init__()

        self.input_dim = in_dim
        self.output_dim = out_dim
        self.epsilon = epsilon
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.W = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))
        self.to(device=self.device)

        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.W) * 0.1
        nn.init.zeros_(self.W)

    def sparsity_loss(self):
        # W = torch.clamp(self.W, min=0, max=1)
        return calc_sparsity_loss(self.W)

    def regularization_loss(self):
        return 0

    def forward(self, X):
        X = X.to(self.device)

        magnitude_log = torch.matmul(torch.log(torch.clamp(torch.abs(X), min=self.epsilon)), self.W)
        magnitude = torch.exp(magnitude_log)
        sign = torch.cos(torch.pi * torch.matmul((X < 0).to(X.dtype), self.W))
        # sign = torch.cos(torch.pi * torch.matmul((X < 0).to(X.dtype), torch.round(self.W)))
        # sign = 1 - 2 * torch.sin(torch.pi * torch.matmul((X < 0).to(X.dtype), torch.round(self.W)))**2

        output = sign * magnitude

        output_hacked = iNPUGradientHack.apply(output, sign, magnitude_log)

        return output_hacked