import torch
import torch.nn as nn
import torch.nn.functional as F

class SuppressionLoss(nn.Module):
    """
    Suppression Loss for controlling the contribution of g_inc branch to base/novel inputs.
    """
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, x_main, indices_base, indices_novel):
        """
        Forward pass of the Suppression loss.
        
        Args:
            x_main (Tensor): Feature outputs from the incremental branch (g_inc), shape [B, D]
            indices_base (Tensor): Boolean tensor indicating base class indices, shape [B]
            indices_novel (Tensor): Boolean tensor indicating novel class indices, shape [B]
        
        Returns:
            Tensor: Suppression loss value
        """
        loss = 0.0
        
        # Calculate suppression loss for base classes (minimize g_inc contribution)
        if indices_base.any():
            loss_base = torch.norm(x_main[indices_base]) / torch.numel(x_main[indices_base])
            loss += loss_base
        
        # Calculate suppression loss for novel classes (maximize g_inc contribution)
        if indices_novel.any():
            loss_novel = -torch.norm(x_main[indices_novel]) / torch.numel(x_main[indices_novel])
            loss += loss_novel
        
        return loss * self.loss_weight
