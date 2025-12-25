import torch
import torch.nn as nn
import torch.nn.functional as F

class SeparationLoss(nn.Module):
    """
    Separation Loss for separating base and novel input-dependent parameters.
    This loss ensures that the feature representations of base and novel classes are well-separated.
    """
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self, params, indices_base, indices_novel, param_avg_dim=0):
        """
        Forward pass of the Separation loss.
        
        Args:
            params (Tensor): Feature tensors from which to calculate separation, shape [B, ...]
            indices_base (Tensor): Boolean tensor indicating base class indices, shape [B]
            indices_novel (Tensor): Boolean tensor indicating novel class indices, shape [B]
            param_avg_dim (int): Dimension along which to average features
        
        Returns:
            Tensor: Separation loss value
        """
        avg_features = []
        
        # Calculate average features for base classes
        if indices_base.any():
            base_params = params[indices_base]
            avg_features.append(
                base_params.mean(param_avg_dim).reshape((1, -1))
            )
        
        # Calculate average features for novel classes
        if indices_novel.any():
            novel_params = params[indices_novel]
            avg_features.append(
                novel_params.mean(param_avg_dim).reshape((1, -1))
            )
        
        if len(avg_features) < 2:
            return torch.tensor(0.0, device=params.device)
        
        # Concatenate average features and normalize
        avg_features = torch.cat(avg_features, dim=0)
        normalized_input = F.normalize(avg_features, dim=-1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(normalized_input, normalized_input.transpose(0, 1))
        
        # Calculate confusion matrix
        confusion_matrix = torch.abs(
            torch.eye(similarity_matrix.shape[0], device=params.device) - similarity_matrix
        )
        
        # Return mean confusion as loss
        loss = torch.mean(confusion_matrix)
        
        return loss * self.loss_weight
