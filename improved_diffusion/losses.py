from torch import nn
from torch.nn import functional as F


class ODEFlowMatchingLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, model, x_0, x_1, t, condition):
        t_coeff = t.unsqueeze(1).unsqueeze(1)
        x_t = t_coeff * x_1 + (1 - t_coeff) * x_0
        noise_pred = model(x_t, t, condition)

        return F.mse_loss(noise_pred, (x_1 - x_0), reduction=self.reduction)
