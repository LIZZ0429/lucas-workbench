import torch
import torch.nn as nn
import torch.nn.functional as F

class DRLoss(nn.Module):
    """
    Dot-Regression Loss for classification.
    L_DR(μ̂_i, ŴETF) = 0.5 * ( ŵ_yi^T μ̂_i - 1 )²
    """
    def __init__(self, reduction='mean', loss_weight=1.0, reg_lambda=0.):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda
    
    def forward(self, feat, target, h_norm2=None, m_norm2=None, avg_factor=None):
        """
        Forward pass of the DR loss.
        
        Args:
            feat (Tensor): Normalized feature embeddings, shape [B, D]
            target (Tensor): Ground truth labels, shape [B, D]
            h_norm2 (Tensor, optional): Norm squared of the feature embeddings
            m_norm2 (Tensor, optional): Norm squared of the ETF vectors
            avg_factor (int, optional): Average factor for loss calculation
        
        Returns:
            Tensor: DR loss value
        """
        # Calculate dot product between feature and target ETF vector
        dot = torch.sum(feat * target, dim=1)
        
        # Default values for h_norm2 and m_norm2
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)
        
        # Compute DR loss according to the original implementation
        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2))**2) / h_norm2)
        
        return loss * self.loss_weight
