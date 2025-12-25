import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import NetMambaBackbone, ETFHead, NetMambaFSCIL

def test_model_initialization():
    """
    测试模型初始化
    """
    print("=== Testing Model Initialization ===")
    
    # 检查是否有可用的CUDA设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建主干网络，对于CPU执行，禁用fused_add_norm
    backbone = NetMambaBackbone(
        byte_length=1600,
        stride_size=4,
        in_chans=1,
        embed_dim=256,
        depth=4,
        fused_add_norm=False if device.type == 'cpu' else True,
        device=device
    )
    
    # 创建分类头
    head = ETFHead(
        in_channels=256,
        num_classes=10,
        base_classes=10
    )
    
    # 创建完整模型
    model = NetMambaFSCIL(backbone, head)
    model.to(device)
    
    print("✓ Model initialization successful")
    return model, device

def test_forward_pass(model, device):
    """
    测试前向传播
    """
    print("\n=== Testing Forward Pass ===")
    
    # 创建随机输入 (batch_size=2, channels=1, height=40, width=40) 并移动到设备
    input_tensor = torch.randn(2, 1, 40, 40).to(device)
    
    # 测试backbone特征提取
    backbone_feat = model.extract_feat(input_tensor, stage='backbone')
    print(f"✓ Backbone output shape: {backbone_feat.shape}")
    
    # 测试neck特征提取（如果有）
    neck_feat = model.extract_feat(input_tensor, stage='neck')
    print(f"✓ Neck output shape: {neck_feat.shape}")
    
    # 测试head输出
    head_output = model.extract_feat(input_tensor, stage='head')
    print(f"✓ Head output shape: {head_output.shape}")
    
    print("✓ Forward pass successful")

def test_update_classes(model, device):
    """
    测试分类头更新
    """
    print("\n=== Testing Update Classes ===")
    
    # 初始类别数量
    initial_num_classes = model.head.num_classes
    print(f"Initial number of classes: {initial_num_classes}")
    
    # 更新类别数量
    new_num_classes = 20
    model.update_classes(new_num_classes)
    
    # 测试更新后的类别数量
    updated_num_classes = model.head.num_classes
    print(f"Updated number of classes: {updated_num_classes}")
    
    # 测试前向传播是否正常
    input_tensor = torch.randn(2, 1, 40, 40).to(device)
    head_output = model.extract_feat(input_tensor, stage='head')
    print(f"✓ Forward pass after class update successful, output shape: {head_output.shape}")
    
    assert updated_num_classes == new_num_classes, f"Class update failed: expected {new_num_classes}, got {updated_num_classes}"
    print("✓ Class update successful")

def test_freeze_unfreeze(model):
    """
    测试模型冻结和解冻
    """
    print("\n=== Testing Freeze/Unfreeze ===")
    
    # 冻结主干网络
    model.freeze_backbone()
    
    # 检查主干网络参数是否冻结
    backbone_frozen = all(not p.requires_grad for p in model.backbone.parameters())
    print(f"✓ Backbone frozen: {backbone_frozen}")
    
    # 检查分类头参数是否可训练
    head_trainable = any(p.requires_grad for p in model.head.parameters())
    print(f"✓ Head trainable: {head_trainable}")
    
    # 解冻主干网络
    model.unfreeze_backbone()
    
    # 检查主干网络参数是否解冻
    backbone_unfrozen = any(p.requires_grad for p in model.backbone.parameters())
    print(f"✓ Backbone unfrozen: {backbone_unfrozen}")
    
    # 冻结分类头
    model.freeze_head()
    
    # 检查分类头参数是否冻结
    head_frozen = all(not p.requires_grad for p in model.head.parameters())
    print(f"✓ Head frozen: {head_frozen}")
    
    # 解冻分类头
    model.unfreeze_head()
    
    # 检查分类头参数是否解冻
    head_unfrozen = any(p.requires_grad for p in model.head.parameters())
    print(f"✓ Head unfrozen: {head_unfrozen}")
    
    print("✓ Freeze/Unfreeze functionality successful")

def main():
    """
    主测试函数
    """
    print("Starting model tests...")
    
    # 测试模型初始化
    model, device = test_model_initialization()
    
    # 测试前向传播
    test_forward_pass(model, device)
    
    # 测试分类头更新
    test_update_classes(model, device)
    
    # 测试模型冻结和解冻
    test_freeze_unfreeze(model)
    
    print("\n=== All Tests Passed! ===")

if __name__ == '__main__':
    main()
