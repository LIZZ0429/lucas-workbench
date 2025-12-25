import os
import sys
import argparse
import torch
import json
from tqdm import tqdm
from mmcv import Config

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import build_model
from dataset import build_dataset
from utils import get_root_logger

def parse_args():
    """
    Parse command line arguments for evaluation.
    """
    parser = argparse.ArgumentParser(description='Evaluate NetMamba-FSCIL on CICIDS2017 dataset')
    
    # Config file
    parser.add_argument('config', help='Evaluation config file path')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    # Evaluation settings
    parser.add_argument('--device', type=str, help='Device used for evaluation')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (only applicable to non-distributed evaluation)')
    parser.add_argument('--batch-size', type=int, help='Batch size for evaluation (overrides config)')
    
    # Evaluation mode
    parser.add_argument('--eval-all', action='store_true', help='Evaluate all tasks after each incremental step')
    parser.add_argument('--out', type=str, default=None, help='Output file path for evaluation results')
    
    return parser.parse_args()

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a single dataloader.
    
    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for evaluation.
        device (torch.device): Device to use for evaluation.
    
    Returns:
        float: Accuracy in percentage.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            # Use simple_test method which returns classification scores
            outputs = model.simple_test(inputs, post_process=False)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

def main():
    """
    Main evaluation function.
    """
    args = parse_args()
    
    # Load configuration file
    cfg = Config.fromfile(args.config)
    
    # Set device
    if args.device is not None:
        device = args.device
    else:
        device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    print(f'Using device: {device}')
    
    # Build model
    print('Building model...')
    model = build_model(cfg.model)
    model.to(device)
    
    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        pretrain_dict = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        # Handle NetMamba's specific checkpoint format
        pretrain_dict = checkpoint['model']
    else:
        pretrain_dict = checkpoint
    
    # Adapt the pretrained weights to our model structure
    # The pretrained model is a base NetMamba, we need to map its weights to our backbone
    model_dict = model.state_dict()
    
    # Create a new state dict that only contains the backbone weights with correct keys
    new_pretrain_dict = {}
    for k, v in pretrain_dict.items():
        # Only take backbone-related weights, ignore decoder weights
        if k.startswith('decoder') or k == 'mask_token':
            continue
        
        # Map the pretrained keys to our backbone keys
        new_k = f'backbone.{k}'
        if new_k in model_dict:
            new_pretrain_dict[new_k] = v
    
    print(f'Loaded {len(new_pretrain_dict)} backbone weights from pretrained model')
    print(f'Ignored {len(pretrain_dict) - len(new_pretrain_dict)} weights')
    
    # Update the model dictionary with the pretrained backbone weights
    model_dict.update(new_pretrain_dict)
    
    # Load the updated state dict, ignoring missing keys for other components
    model.load_state_dict(model_dict, strict=False)
    
    # Build dataset and dataloader
    print('Building dataset...')
    test_dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    
    # Use batch size from command line if provided, otherwise use config
    batch_size = args.batch_size if args.batch_size is not None else cfg.data.samples_per_gpu
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.workers_per_gpu,
        pin_memory=True
    )
    
    # Evaluate model
    print('Starting evaluation...')
    accuracy = evaluate_model(model, test_dataloader, device)
    
    # Print results
    print(f'\n=== Evaluation Results ===')
    print(f'Accuracy: {accuracy:.2f}%')
    
    # Prepare evaluation results
    eval_results = {
        'accuracy': accuracy,
        'config': cfg.filename,
        'checkpoint': args.checkpoint,
        'dataset': 'CICIDS2017',
        'batch_size': batch_size,
        'num_samples': len(test_dataset)
    }
    
    # Save evaluation results if output path is provided
    if args.out is not None:
        output_dir = os.path.dirname(args.out)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.out, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f'\nEvaluation results saved to: {args.out}')
    
    print('\n=== Evaluation completed ===')

if __name__ == '__main__':
    main()
