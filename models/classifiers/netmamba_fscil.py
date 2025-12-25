import torch
import torch.nn as nn
import torch.nn.functional as F

from ..losses.suppression_loss import SuppressionLoss
from ..losses.separation_loss import SeparationLoss
from ..memory.memory import MemoryModule

class NetMambaFSCIL(nn.Module):
    """
    NetMamba-FSCIL: A Mamba-based Few-Shot Class-Incremental Learning model for intrusion detection.
    """
    def __init__(self, backbone, head, neck=None, pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.neck = neck
        
        # Initialize loss functions
        self.suppression_loss = SuppressionLoss(loss_weight=1.0)
        self.separation_loss = SeparationLoss(loss_weight=1.0)
        
        # Initialize Memory Module
        # Get feature dimension from neck output
        self.memory = None
        if neck:
            feature_dim = neck.out_channels
            self.memory = MemoryModule(feature_dim)
        
        if pretrained is not None:
            self.load_pretrained(pretrained)
    
    def load_pretrained(self, pretrained):
        """
        Load pretrained weights from a checkpoint.
        """
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load backbone weights
        backbone_state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
        self.backbone.load_state_dict(backbone_state_dict, strict=False)
        
        print(f"Loaded pretrained weights from {pretrained}")
    
    def extract_feat(self, img, stage='head'):
        """
        Extract features from different stages of the model.
        """
        assert stage in ['backbone', 'neck', 'head'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '\
             '"neck" and "head"')
        
        # Extract features from backbone
        x = self.backbone(img)
        
        if stage == 'backbone':
            return x
        
        # Pass through neck if exists
        if self.neck is not None:
            x = self.neck(x)
        
        if stage == 'neck':
            return x
        
        # Pass through head
        # Get the actual feature tensor from the dictionary if needed
        feat = x['out'] if isinstance(x, dict) else x
        x = self.head(feat)
        
        return x
    
    def forward_train(self, img, gt_label):
        """
        Forward pass during training.
        """
        # Extract features up to neck (Projector)
        x = self.extract_feat(img, stage='neck')
        
        # Get losses from head
        losses = self.head.forward_train(x['out'], gt_label)
        
        # Calculate base/novel class indices
        base_classes = getattr(self.head, 'base_classes', 0)
        indices_base = gt_label < base_classes
        indices_novel = gt_label >= base_classes
        
        # Calculate Suppression Loss if both base and novel classes are present
        if self.neck and hasattr(self.neck, 'use_new_branch') and self.neck.use_new_branch:
            if indices_base.any() or indices_novel.any():
                loss_supp = self.suppression_loss(x['main'], indices_base, indices_novel)
                losses['loss_suppression'] = loss_supp
        
        # Calculate Separation Loss if needed
        # This would require access to the SSM parameters (dts, Bs, Cs, etc.)
        # which are not currently returned by the Projector
        # We'll need to update the Projector to return these parameters if needed
        
        return losses
    
    def simple_test(self, img, post_process=True):
        """
        Simple test without augmentation.
        """
        x = self.extract_feat(img, stage='neck')
        return self.head.simple_test(x['out'], post_process=post_process)
    
    def update_memory(self, dataloader):
        """
        Update the memory with class centers computed from the given dataloader.
        
        Args:
            dataloader (DataLoader): Dataloader to iterate over data.
        """
        if self.memory:
            self.memory.compute_class_centers(self, dataloader)
    
    def get_memory_class_centers(self):
        """
        Get all class centers from memory.
        
        Returns:
            Tensor: All class centers, shape [num_classes, feature_dim].
        """
        if self.memory:
            return self.memory.get_all_class_centers()
        return None
    
    def save_memory(self, path):
        """
        Save the memory to a file.
        
        Args:
            path (str): Path to save the memory.
        """
        if self.memory:
            self.memory.save(path)
    
    def load_memory(self, path):
        """
        Load the memory from a file.
        
        Args:
            path (str): Path to load the memory from.
        """
        if self.memory:
            self.memory.load(path)
    
    def reset_memory(self):
        """
        Reset the memory.
        """
        if self.memory:
            self.memory.reset()
    
    def update_classes(self, new_num_classes):
        """
        Update the number of classes in the head for incremental learning.
        """
        if hasattr(self.head, 'update_classes'):
            self.head.update_classes(new_num_classes)
        else:
            raise NotImplementedError("Head does not support updating classes")
    
    def freeze_backbone(self):
        """
        Freeze the backbone parameters for incremental learning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """
        Unfreeze the backbone parameters.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def freeze_head(self):
        """
        Freeze the head parameters.
        """
        for param in self.head.parameters():
            param.requires_grad = False
    
    def unfreeze_head(self):
        """
        Unfreeze the head parameters.
        """
        for param in self.head.parameters():
            param.requires_grad = True
