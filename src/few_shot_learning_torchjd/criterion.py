import random

import torch
import torch.nn as nn


class RegularizedLoss(nn.Module):
    def __init__(self, 
                 l1_weight: float,
                 l2_weight: float,
                 mse_weight: float,
                 kld_weight: float,
                 n_counterfactuals: int,
                 #l2_weight: float,
                 #lambda_hessian: float
                 ):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.mse_weight = mse_weight
        self.kld_weight = kld_weight
        self.n_counterfactuals = n_counterfactuals
        #self.l2_weight = l2_weight
        #self.lambda_hessian = lambda_hessian
        self.loss = nn.MSELoss(reduction="none")
    
    def forward(self, model: nn.Module,
                model_input: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        output, mu, logvar = model(model_input)

        mse_loss = self.loss(output, torch.ones_like(output))
        if self.n_counterfactuals > 0:
            counterfactuals = self._create_counterfactuals(model_input=model_input)
            counterfactual_output, _, _ = model(counterfactuals)
            counterfactual_loss = self.loss(counterfactual_output, torch.zeros_like(counterfactual_output))
            mse_loss = torch.cat([mse_loss, counterfactual_loss], dim=0)

        scaled_mse_loss = self.mse_weight * mse_loss

        if self.l1_weight > 0:
            l1_loss = torch.stack([p.abs().mean() for p in model.parameters() if p.requires_grad]).mean()
        else:
            l1_loss = torch.Tensor([0])
        scaled_l1_loss = self.l1_weight * l1_loss

        if self.l2_weight > 0:
            l2_loss = torch.stack([p.pow(2).mean() for p in model.parameters() if p.requires_grad]).mean()
        else:
            l2_loss = torch.Tensor([0])
        scaled_l2_loss = self.l2_weight * l2_loss

        if self.kld_weight > 0:
            kld_loss = self._kld_loss(mu, logvar)
        else:
            kld_loss = torch.Tensor([0])

        scaled_kld_loss = self.kld_weight * kld_loss

        # l2_loss = torch.stack([p.pow(2).mean() for p in model.parameters() if p.requires_grad]).mean()
        # scaled_l2_loss = self.l2_weight * l2_loss

        total_loss = scaled_mse_loss + scaled_l1_loss + scaled_l2_loss + scaled_kld_loss

        return total_loss, mse_loss, l1_loss, kld_loss
    
    def _kld_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Calculate the Kullback-Leibler divergence loss."""
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
        return kld_loss
    
    def _create_counterfactuals(self, model_input: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:

        images, labels = model_input
        counterfactuals = []
        random_rolls = random.sample(range(1, 10), k=self.n_counterfactuals)
        for roll in random_rolls:
            counterfactual = labels.roll(roll, dims=-1) 
            counterfactuals.append(counterfactual)
        return (torch.cat([images]*self.n_counterfactuals, dim=0), torch.cat(counterfactuals, dim=0))

