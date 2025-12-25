from .backbones import *
from .heads import *
from .necks import *
from .classifiers import *

__all__ = [
    # Backbones
    'NetMambaBackbone',
    'net_mamba_backbone',
    'net_mamba_backbone_small',
    'net_mamba_backbone_large',
    
    # Necks
    'DualSelectiveSSMProjector',
    
    # Heads
    'ETFHead',
    
    # Classifiers
    'NetMambaFSCIL',
    
    # Builders
    'build_model',
]


def build_model(cfg):
    """
    Build a model from config dict.
    
    Args:
        cfg (dict): Config dict.
        
    Returns:
        nn.Module: The constructed model.
    """
    model_type = cfg.pop('type')
    
    if model_type == 'NetMambaFSCIL':
        # Build backbone
        backbone_cfg = cfg.pop('backbone')
        backbone = eval(backbone_cfg.pop('type'))(**backbone_cfg)
        
        # Build neck if exists
        neck = None
        if 'neck' in cfg:
            neck_cfg = cfg.pop('neck')
            neck = eval(neck_cfg.pop('type'))(**neck_cfg)
        
        # Build head
        head_cfg = cfg.pop('head')
        head = eval(head_cfg.pop('type'))(**head_cfg)
        
        # Build classifier
        return NetMambaFSCIL(backbone, head, neck, **cfg)
    else:
        # Directly build other model types
        return eval(model_type)(**cfg)
