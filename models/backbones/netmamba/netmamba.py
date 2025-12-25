import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .modules import StrideEmbed, create_block, RMSNorm, rms_norm_fn
from timm.models.layers import trunc_normal_, lecun_normal_
import math
from functools import partial

class NetMamba(nn.Module):
    def __init__(self, byte_length=1600, stride_size=4, in_chans=1,
                 embed_dim=192, depth=4,
                 num_classes=1000,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 rms_norm=True,
                 fused_add_norm=True,
                 device=None, dtype=None,
                 **kwargs):
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.is_pretrain = False
        self.stride_size = stride_size

        # --------------------------------------------------------------------------
        # NetMamba encoder specifics
        self.patch_embed = StrideEmbed(byte_length, stride_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_cls_token = 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + num_cls_token, embed_dim))
        # Mamba blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.ModuleList([
            create_block(
                embed_dim,
                ssm_cfg=None,
                norm_epsilon=1e-5,
                rms_norm=rms_norm,
                residual_in_fp32=True,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
                drop_path=inter_dpr[i],
            )  for i in range(depth)])
        self.norm_f = RMSNorm(embed_dim, eps=1e-5) if rms_norm else nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------

        # Initialize weights
        self.patch_embed.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(partial(self._init_weights, n_layer=depth))
    
    def _init_weights(
        self, module, n_layer=None,
        initializer_range=0.02,
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,
    ):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            lecun_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.ones_(module.weight)

        if rescale_prenorm_residual and n_layer is not None:
            if isinstance(module, nn.Linear) and module.bias is not None:
                if module.bias.shape[0] == self.embed_dim:
                    nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                    with torch.no_grad():
                        module.weight /= math.sqrt(n_residuals_per_layer * n_layer)
    
    def forward_encoder(self, x, if_mask=False):
        """
        x: [B, 1, H, W] or [B, 1, L] where L = H*W
        return: [B, seq_len, embed_dim]
        """
        if x.dim() == 4:
            # Input is [B, C, H, W]
            B, C, H, W = x.shape
            assert C == 1, "Input images should be grayscale"
            
            # Reshape to [B, C, L] where L = H*W
            x = x.reshape(B, C, -1)
        elif x.dim() == 3:
            # Input is already [B, C, L]
            B, C, L = x.shape
            assert C == 1, "Input channel should be 1"
        else:
            raise ValueError(f"Input tensor should be 3D or 4D, got {x.dim()}D")
        
        # Embed patches
        x = self.patch_embed(x)
        
        # Add pos embed without cls token
        x = x + self.pos_embed[:, :-1, :]
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, -1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)
        x = self.pos_drop(x)
        
        # Apply Mamba blocks
        residual = None
        hidden_states = x
        for blk in self.blocks:
            hidden_states, residual = blk(hidden_states, residual)
        
        # Final norm
        if hasattr(self, 'fused_add_norm') and self.fused_add_norm:
            fused_add_norm_fn = rms_norm_fn
            x = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=True,
            )
        else:
            x = self.norm_f(hidden_states + (residual if residual is not None else 0))
        
        # Return feature sequence (without cls token for now)
        # return x[:, :-1, :]  # [B, seq_len, embed_dim]
        return x  # [B, seq_len + 1, embed_dim] including cls token
    
    def forward(self, x):
        """
        x: [B, 1, H, W] or [B, 1, L]
        return: [B, seq_len + 1, embed_dim] including cls token
        """
        return self.forward_encoder(x, if_mask=False)


def net_mamba_backbone(**kwargs):
    model = NetMamba(
        is_pretrain=False, stride_size=4, embed_dim=256, depth=4,
        **kwargs)
    return model


def net_mamba_backbone_small(**kwargs):
    model = NetMamba(
        is_pretrain=False, stride_size=4, embed_dim=192, depth=3,
        **kwargs)
    return model


def net_mamba_backbone_large(**kwargs):
    model = NetMamba(
        is_pretrain=False, stride_size=4, embed_dim=384, depth=6,
        **kwargs)
    return model
