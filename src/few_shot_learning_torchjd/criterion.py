import torch
import torch.nn as nn


class RegularizedLoss(nn.Module):
    def __init__(self, l1_weight: float,
                 mse_weight: float,
                 kld_weight: float,
                 #l2_weight: float,
                 #lambda_hessian: float
                 ):
        super().__init__()
        self.l1_weight = l1_weight
        self.mse_weight = mse_weight
        self.kdl_weight = kld_weight
        #self.l2_weight = l2_weight
        #self.lambda_hessian = lambda_hessian
        self.loss = nn.MSELoss(reduction="none")
    
    def forward(self, model: nn.Module,
                model_output: torch.Tensor,
                mu: torch.Tensor,
                logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        mse_loss = self.loss(model_output, torch.ones_like(model_output))
        scaled_mse_loss = self.mse_weight * mse_loss

        l1_loss = torch.stack([p.abs().mean() for p in model.parameters() if p.requires_grad]).mean()
        scaled_l1_loss = self.l1_weight * l1_loss

        kld_loss = self._kld_loss(mu, logvar)
        scaled_kld_loss = self.kdl_weight * kld_loss

        # l2_loss = torch.stack([p.pow(2).mean() for p in model.parameters() if p.requires_grad]).mean()
        # scaled_l2_loss = self.l2_weight * l2_loss

        total_loss = scaled_mse_loss + scaled_l1_loss + scaled_kld_loss

        return total_loss, mse_loss, l1_loss, kld_loss
    
    def _kld_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Calculate the Kullback-Leibler divergence loss."""
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        return kld_loss