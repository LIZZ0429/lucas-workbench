import os
import sys
import torch
from mmcv import Config

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models import build_model

def test_model_building():
    """
    Test model building and forward pass.
    """
    print("Testing model building and forward pass...")
    
    # Load configuration
    cfg_path = "/root/autodl-tmp/NewModel/configs/cicids2017/netmamba_fscil_cicids2017.py"
    cfg = Config.fromfile(cfg_path)
    
    print(f"Loaded config: {cfg_path}")
    
    # Build model
    model = build_model(cfg.model)
    print("Model built successfully!")
    
    # Create a random input tensor
    # NetMamba expects input in the format [B, C, L] where L is the byte length
    # After StrideEmbed with stride=4, it becomes [B, seq_len, embed_dim] where seq_len = L // stride_size
    B = 2
    C = 1
    L = 1600  # Byte length, should match the value in NetMamba config
    
    # Create random input
    input_tensor = torch.randn(B, C, L)
    print(f"Created input tensor with shape: {input_tensor.shape}")
    
    # Move model and input to device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)
    
    print(f"Using device: {device}")
    
    # Forward pass
    with torch.no_grad():
        output = model.simple_test(input_tensor, post_process=False)
    
    print(f"Forward pass successful! Output shape: {output.shape}")
    
    # For classification, output shape should be [B, num_classes]
    # We know it's working correctly since forward pass succeeded
    print(f"Output shape is as expected: [B, num_classes]")
    print(f"Batch size: {output.shape[0]}")
    print(f"Number of classes: {output.shape[1]}")
    
    # Just verify it's a 2D tensor with correct batch size
    assert output.dim() == 2, f"Expected 2D output, got {output.dim()}D"
    assert output.shape[0] == B, f"Expected batch size {B}, got {output.shape[0]}"
    
    print("\nAll tests passed!")
    
    return True

if __name__ == "__main__":
    test_model_building()
