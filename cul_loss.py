import torch
import torch.nn as nn


class CustomLossWithRegularization(nn.Module):
    def __init__(self, lambda_reg):
        super(CustomLossWithRegularization, self).__init__()
        self.lambda_reg = lambda_reg
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true, model):
        bce_loss = self.bce_loss(y_pred, y_true)

        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)

        total_loss = bce_loss + self.lambda_reg * l2_reg

        return total_loss
