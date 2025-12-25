import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ..losses.dr_loss import DRLoss

class ETFHead(nn.Module):
    """
    Enhanced Topological Feature (ETF) Head for incremental learning.
    This head uses ETF vectors and DR Loss for classification.
    """
    def __init__(self, in_channels, num_classes, base_classes=60, with_bn=False, with_avg_pool=False, init_cfg=None):
        super().__init__()
        # Evaluation classes
        self.eval_classes = num_classes
        self.base_classes = base_classes
        
        # Training settings about different length for different classes
        self.with_len = False
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.with_bn = with_bn
        self.with_avg_pool = with_avg_pool
        
        # Calculate accuracy for top-1
        self.topk = (1,)
        self.cal_acc = True
        
        # Batch normalization if needed
        if with_bn:
            self.bn = nn.BatchNorm1d(in_channels)
        
        # Generate ETF vectors
        self._generate_etf_vectors()
        
        # Initialize DR Loss
        self.dr_loss = DRLoss(loss_weight=1.0)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(p=0.7)
    
    def _generate_etf_vectors(self):
        """
        Generate ETF vectors using orthogonal matrix projection.
        """
        # Generate random orthogonal matrix
        rand_mat = np.random.random(size=(self.in_channels, self.num_classes))
        orth_vec, _ = np.linalg.qr(rand_mat)
        orth_vec = torch.tensor(orth_vec).float()
        
        # Create ETF vectors
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        etf_vec = torch.mul(
            torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
            math.sqrt(self.num_classes / (self.num_classes - 1))
        )
        
        # Register ETF vectors as buffer
        self.register_buffer('etf_vec', etf_vec)
        
        # Create ETF rect for training settings
        etf_rect = torch.ones((1, self.num_classes), dtype=torch.float32)
        self.etf_rect = etf_rect
    
    def pre_logits(self, x):
        """
        Pre-logits processing.
        
        Args:
            x (Tensor): Input features, shape [B, in_channels]
        
        Returns:
            Tensor: Processed features, shape [B, in_channels]
        """
        if isinstance(x, dict):
            x = x['out']
        
        # Normalize features
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        
        return x
    
    def forward(self, x):
        """
        Forward pass of the ETF head.
        
        Args:
            x (Tensor): Input features from Projector, shape [B, out_channels]
        
        Returns:
            Tensor: Classification scores, shape [B, num_classes]
        """
        if self.with_avg_pool:
            if x.dim() == 4:
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = x.view(x.size(0), -1)
        
        if self.with_bn:
            x = self.bn(x)
        
        # Normalize input features
        x = self.pre_logits(x)
        
        # Calculate classification scores using ETF vectors
        cls_score = x @ self.etf_vec
        
        return cls_score
    
    def forward_train(self, x, gt_label):
        """
        Forward pass during training.
        
        Args:
            x (Tensor): Input features from Projector, shape [B, out_channels]
            gt_label (Tensor): Ground truth labels, shape [B]
        
        Returns:
            dict: Losses dictionary
        """
        if isinstance(x, dict):
            x = x['out']
        
        # Normalize input features
        x = self.pre_logits(x)
        
        # Apply dropout for regularization during training
        x = self.dropout(x)
        
        # Get ETF vectors for ground truth labels
        if self.with_len:
            etf_vec = self.etf_vec * self.etf_rect.to(device=self.etf_vec.device)
            target = (etf_vec * self.produce_training_rect(gt_label, self.num_classes))[:, gt_label].t()
        else:
            target = self.etf_vec[:, gt_label].t()
        
        # Calculate DR loss
        losses = self.loss(x, target)
        
        # Calculate accuracy if needed
        if self.cal_acc:
            with torch.no_grad():
                cls_score = x @ self.etf_vec
                acc = self.compute_accuracy(cls_score[:, :self.eval_classes], gt_label)
                assert len(acc) == len(self.topk)
                losses['accuracy'] = {
                    f'top-{k}': a
                    for k, a in zip(self.topk, acc)
                }
        
        return losses
    
    def loss(self, feat, target, **kwargs):
        """
        Compute the loss for incremental learning.
        
        Args:
            feat (Tensor): Normalized features from Projector, shape [B, in_channels]
            target (Tensor): Target ETF vectors for ground truth labels, shape [B, in_channels]
        
        Returns:
            dict: Losses dictionary
        """
        losses = {}
        
        # Compute DR loss
        if self.with_len:
            loss_dr = self.dr_loss(feat, target, m_norm2=torch.norm(target, p=2, dim=1))
        else:
            loss_dr = self.dr_loss(feat, target)
        losses['loss_dr'] = loss_dr
        
        # Total loss
        losses['loss'] = loss_dr
        
        return losses
    
    def simple_test(self, x, softmax=False, post_process=False):
        """
        Simple test without augmentation.
        
        Args:
            x (Tensor): Input features, shape [B, out_channels]
            softmax (bool): Whether to apply softmax (should be False for original implementation)
            post_process (bool): Whether to apply post-processing
        
        Returns:
            Tensor: Classification scores
        """
        if isinstance(x, dict):
            x = x['out']
        
        x = self.pre_logits(x)
        cls_score = x @ self.etf_vec
        cls_score = cls_score[:, :self.eval_classes]
        
        assert not softmax, "Original implementation does not use softmax in simple_test"
        
        return cls_score
    
    def compute_accuracy(self, cls_score, gt_labels):
        """
        Compute the accuracy.
        
        Args:
            cls_score (Tensor): Classification scores, shape [B, num_classes]
            gt_labels (Tensor): Ground truth labels, shape [B]
        
        Returns:
            list: Accuracy for each topk
        """
        acc = []
        for k in self.topk:
            _, pred = cls_score.topk(k, 1, True, True)
            correct = pred.eq(gt_labels.view(-1, 1).expand_as(pred))
            correct_k = correct.view(-1).float().sum(0, keepdim=True)
            acc.append(correct_k.mul_(100.0 / cls_score.size(0)))
        return acc
    
    def update_classes(self, new_num_classes):
        """
        Update the number of classes when adding new classes in incremental learning.
        
        Args:
            new_num_classes (int): New number of classes
        """
        if new_num_classes <= self.num_classes:
            return
        
        # Regenerate ETF vectors with new number of classes
        self.num_classes = new_num_classes
        self.eval_classes = new_num_classes
        self._generate_etf_vectors()
    
    @staticmethod
    def produce_training_rect(label, num_classes):
        """
        Produce training rectangle for different classes.
        
        Args:
            label (Tensor): Ground truth labels, shape [B]
            num_classes (int): Total number of classes
        
        Returns:
            Tensor: Training rectangle, shape [1, num_classes]
        """
        uni_label, count = torch.unique(label, return_counts=True)
        batch_size = label.size(0)
        uni_label_num = uni_label.size(0)
        
        gamma = torch.tensor(batch_size / uni_label_num, device=label.device, dtype=torch.float32)
        rect = torch.ones(1, num_classes).to(device=label.device, dtype=torch.float32)
        rect[0, uni_label] = torch.sqrt(gamma / count)
        
        return rect