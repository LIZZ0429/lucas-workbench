from collections import OrderedDict

import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from timm.models.layers import trunc_normal_

# 使用NetMamba的1D MambaBlock
from ..backbones.netmamba.modules import create_block

class DualSelectiveSSMProjector(BaseModule):
    """Dual selective SSM branch in Mamba-FSCIL framework.

        This module integrates our dual selective SSM branch for dynamic adaptation in few-shot
        class-incremental learning tasks.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of intermediate channels in MLP projections, defaults to twice the in_channels if not specified.
            d_state (int): Dimension of the hidden state in the SSM.
            d_rank (int, optional): Dimension rank in the SSM, if not provided, defaults to d_state.
            ssm_expand_ratio (float): Expansion ratio for the SSM block.
            num_layers (int): Number of layers in the MLP projections.
            num_layers_new (int, optional): Number of layers in the new branch MLP projections, defaults to num_layers if not specified.
            feat_size (int): Size of the input feature map.
            use_new_branch (bool): If True, uses an additional branch for incremental learning.
            loss_weight_supp (float): Loss weight for suppression term for base classes.
            loss_weight_supp_novel (float): Loss weight for suppression term for novel classes.
            loss_weight_sep (float): Loss weight for separation term during the base session.
            loss_weight_sep_new (float): Loss weight for separation term during the incremental session.
            param_avg_dim (str): Dimensions to average for computing averaged input-dependment parameter features.
            detach_residual (bool): If True, detaches the residual connections during the output computation.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, 
                 d_state=256, d_rank=None, ssm_expand_ratio=1, num_layers=2,
                 num_layers_new=None, feat_size=2, use_new_branch=True,
                 loss_weight_supp=0.0, loss_weight_supp_novel=0.0,
                 loss_weight_sep=0.0, loss_weight_sep_new=0.0,
                 param_avg_dim='0-1', detach_residual=False):
        super(DualSelectiveSSMProjector, self).__init__(init_cfg=None)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = in_channels * 2 if mid_channels is None else mid_channels
        self.feat_size = feat_size
        self.d_state = d_state
        self.d_rank = d_rank if d_rank is not None else d_state
        self.use_new_branch = use_new_branch
        self.num_layers = num_layers
        self.num_layers_new = self.num_layers if num_layers_new is None else num_layers_new
        self.detach_residual = detach_residual
        self.loss_weight_supp = loss_weight_supp
        self.loss_weight_supp_novel = loss_weight_supp_novel
        self.loss_weight_sep = loss_weight_sep
        self.loss_weight_sep_new = loss_weight_sep_new
        self.param_avg_dim = [int(item) for item in param_avg_dim.split('-')]

        # Positional embeddings for features
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.feat_size * self.feat_size, out_channels))
        trunc_normal_(self.pos_embed, std=.02)

        if self.use_new_branch:
            self.pos_embed_new = nn.Parameter(
                torch.zeros(1, self.feat_size * self.feat_size, out_channels))
            trunc_normal_(self.pos_embed_new, std=.02)

        # MLP projections for main branch
        if self.num_layers == 3:
            self.mlp_proj = self.build_mlp(in_channels, out_channels, self.mid_channels, num_layers=3)
        elif self.num_layers == 2:
            self.mlp_proj = self.build_mlp(in_channels, out_channels, self.mid_channels, num_layers=2)

        # 1D Mamba block for main branch (g_base)
        self.block = create_block(
            out_channels,
            ssm_cfg={'expand': ssm_expand_ratio, 'd_state': d_state, 'dt_rank': self.d_rank},
            norm_epsilon=1e-5,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=False,
            layer_idx=0,
            drop_path=0.0
        )

        # MLP projections for new branch if needed
        if self.use_new_branch:
            if self.num_layers_new == 3:
                self.mlp_proj_new = self.build_mlp(in_channels, out_channels, self.mid_channels, num_layers=3)
            elif self.num_layers_new == 2:
                self.mlp_proj_new = self.build_mlp(in_channels, out_channels, self.mid_channels, num_layers=2)

            # 1D Mamba block for new branch (g_inc)
            self.block_new = create_block(
                out_channels,
                ssm_cfg={'expand': ssm_expand_ratio, 'd_state': d_state, 'dt_rank': self.d_rank},
                norm_epsilon=1e-5,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=False,
                layer_idx=1,
                drop_path=0.0
            )

        # Identity projection for residual connection
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.use_residual_proj = False

        self.init_weights()

    def build_mlp(self, in_channels, out_channels, mid_channels, num_layers):
        """Builds the MLP projection part of the neck.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int): Number of mid-level channels.
            num_layers (int): Number of linear layers in the MLP.

        Returns:
            nn.Sequential: The MLP layers as a sequential module.
        """
        layers = []
        layers.append(
            nn.Linear(in_channels, mid_channels, bias=True)
        )
        layers.append(nn.LeakyReLU(0.1))

        if num_layers == 3:
            layers.append(
                nn.Linear(mid_channels, mid_channels, bias=True)
            )
            layers.append(nn.LeakyReLU(0.1))

        layers.append(
            nn.Linear(mid_channels, out_channels, bias=False)
        )
        return nn.Sequential(*layers)

    def init_weights(self):
        """
        Initialize the weights of the projector.
        For NetMamba Block, we don't need to zero out any weights.
        """
        # NetMamba Block doesn't have in_proj attribute, so we skip this initialization
        pass

    def forward(self, x):
        """Forward pass for DualSelectiveSSMProjector, integrating both the main and an optional new branch for processing.

            Args:
                x (Tensor): Input tensor, shape [B, seq_len, embed_dim] from NetMamba.

            Returns:
                dict: A dictionary of outputs including processed features from main and new branches,
                      along with the combined final output.
            """
        # Extract the last element if input is a tuple (from previous layers).
        if isinstance(x, tuple):
            x = x[-1]
        
        B, seq_len, C = x.shape
        identity = x
        outputs = {}

        C, dts, Bs, Cs, C_new, dts_new, Bs_new, Cs_new = None, None, None, None, None, None, None, None

        if self.detach_residual:
            self.block.eval()
            self.mlp_proj.eval()

        # Prepare the identity projection for the residual connection
        # For NetMamba output [B, seq_len+1, embed_dim], we need to project to out_channels
        # Take the last token (cls token) and project it to out_channels dimension
        cls_token = identity[:, -1:, :]  # [B, 1, embed_dim]
        identity_proj = self.mlp_proj(cls_token).squeeze(1)  # [B, out_channels]
        
        # Process the input tensor through MLP projection and add positional embeddings
        x = self.mlp_proj(identity)
        x = x + self.pos_embed[:, :seq_len, :]

        # First selective SSM branch processing (g_base)
        residual = None
        x, residual = self.block(x, residual=residual)
        
        # Average pooling to get [B, out_channels]
        x = x.mean(dim=1)  # [B, out_channels]

        # New branch processing for incremental learning sessions, if enabled.
        if self.use_new_branch:
            x_new = self.mlp_proj_new(identity.detach())
            x_new += self.pos_embed_new[:, :seq_len, :]
            
            # Incremental selective SSM branch processing (g_inc)
            residual_new = None
            x_new, residual_new = self.block_new(x_new, residual=residual_new)
            
            # Average pooling to get [B, out_channels]
            x_new = x_new.mean(dim=1)  # [B, out_channels]

        """Combines outputs from the main and new branches with the identity projection."""
        if not self.use_new_branch:
            outputs['main'] = C if C is not None else x
            outputs['residual'] = identity_proj
            x = x + identity_proj
        else:
            outputs['main'] = C_new if C_new is not None else x_new
            outputs['residual'] = x + identity_proj
            if self.detach_residual:
                x = x.detach() + identity_proj.detach() + x_new
            else:
                x = x + identity_proj + x_new

        outputs['out'] = x
        return outputs

    def build_mlp(self, in_channels, out_channels, mid_channels, num_layers):
        """Builds the MLP projection part of the neck."""
        layers = []
        layers.append(nn.Linear(in_channels, mid_channels, bias=True))
        layers.append(nn.LeakyReLU(0.1))

        if num_layers == 3:
            layers.append(nn.Linear(mid_channels, mid_channels, bias=True))
            layers.append(nn.LeakyReLU(0.1))

        layers.append(nn.Linear(mid_channels, out_channels, bias=False))
        return nn.Sequential(*layers)
