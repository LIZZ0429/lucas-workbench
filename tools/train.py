import os
import sys
import argparse
import random
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, build_runner, build_optimizer

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import build_model
from dataset import build_dataset
from utils import get_root_logger, set_random_seed

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Train NetMamba-FSCIL on CICIDS2017 dataset')
    
    # Config file
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='The dir to save logs and models')
    parser.add_argument('--resume-from', help='The checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='Whether not to evaluate the checkpoint during training')
    
    # GPU settings
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--device', help='Device used for training.')
    group_gpus.add_argument('--gpu-id', type=int, default=0, help='ID of gpu to use (only applicable to non-distributed training)')
    
    # Seed settings
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--diff-seed', action='store_true', help='Whether or not set different seeds for different ranks')
    parser.add_argument('--deterministic', action='store_true', help='Whether to set deterministic options for CUDNN backend')
    
    # Config override
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='Override some settings in the used config')
    
    # Distributed training
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='Job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    return parser.parse_args()

def main():
    """
    Main training function.
    """
    args = parse_args()
    
    # Load configuration file
    cfg = Config.fromfile(args.config)
    
    # Merge config with command line options
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    # Set work directory
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = os.path.join('./work_dirs', os.path.splitext(os.path.basename(args.config))[0])
    
    # Create work directory
    os.makedirs(cfg.work_dir, exist_ok=True)
    
    # Dump config without yapf formatting (avoiding verify parameter issue)
    with open(os.path.join(cfg.work_dir, os.path.basename(args.config)), 'w') as f:
        f.write(cfg.text)
    
    # Set device
    if args.device is not None:
        cfg.device = args.device
    else:
        cfg.device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    
    # Initialize distributed training
    distributed = args.launcher != 'none'
    if distributed:
        init_dist(args.launcher, **cfg.dist_params)
    
    # Initialize logger
    logger = get_root_logger(log_file=os.path.join(cfg.work_dir, 'train.log'), log_level=cfg.log_level)
    logger.info(f'Using device: {cfg.device}')
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.text}')
    
    # Set random seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    if distributed and args.diff_seed:
        seed += dist.get_rank()
    logger.info(f'Setting random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    
    # Build model
    logger.info('Building model...')
    model = build_model(cfg.model)
    model.to(cfg.device)
    
    # Build dataset and dataloaders
    logger.info('Building dataset...')
    train_dataset = build_dataset(cfg.data.train)
    
    # Create dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.data.samples_per_gpu,
        shuffle=True,
        num_workers=cfg.data.workers_per_gpu,
        pin_memory=True
    )
    
    # Build validation dataset and dataloader if needed
    val_dataloader = None
    if not args.no_validate and hasattr(cfg.data, 'val'):
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.data.samples_per_gpu,
            shuffle=False,
            num_workers=cfg.data.workers_per_gpu,
            pin_memory=True
        )
    
    # Build optimizer
    logger.info('Building optimizer...')
    optimizer = build_optimizer(model, cfg.optimizer)
    
    # Load checkpoint if needed
    if cfg.load_from:
        logger.info(f'Loading checkpoint from {cfg.load_from}')
        checkpoint = torch.load(cfg.load_from, map_location=cfg.device)
        
        # Adapt the pretrained weights to our model structure
        pretrain_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
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
        
        logger.info(f'Loaded {len(new_pretrain_dict)} backbone weights from pretrained model')
        logger.info(f'Ignored {len(pretrain_dict) - len(new_pretrain_dict)} weights')
        
        # Update the model dictionary with the pretrained backbone weights
        model_dict.update(new_pretrain_dict)
        
        # Load the updated state dict, ignoring missing keys for other components
        model.load_state_dict(model_dict, strict=False)
    
    # Simple training loop since we're using a custom model architecture
    logger.info('Starting training...')
    
    # Set model to training mode
    model.train()
    
    # Initialize variables for training
    best_accuracy = 0.0
    epochs = cfg.runner.max_epochs
    
    for epoch in range(epochs):
        logger.info(f'Starting epoch {epoch+1}/{epochs}')
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(cfg.device)
            labels = labels.to(cfg.device)
            
            # Forward pass using the correct method for training
            losses = model.forward_train(inputs, labels)
            
            # Get the total loss by summing only loss values (excluding logits and accuracy dict)
            loss_values = {k: v for k, v in losses.items() if k != 'logits' and not isinstance(v, dict)}
            loss = sum(loss_values.values())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Note: The head.forward_train should return logits in the losses dictionary
            # Let's check the actual structure by printing it once
            if batch_idx == 0:
                logger.info(f"Losses dictionary structure: {list(losses.keys())}")
            
            # Calculate accuracy if logits are available
            if 'logits' in losses:
                _, predicted = losses['logits'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            elif 'accuracy' in losses:
                # If accuracy is already calculated as a dict
                if isinstance(losses['accuracy'], dict):
                    # Extract top-1 accuracy from the dict
                    if 'top-1' in losses['accuracy']:
                        correct += int(losses['accuracy']['top-1'] * labels.size(0) / 100)  # Convert percentage to decimal
                elif isinstance(losses['accuracy'], (float, torch.Tensor)):
                    # If accuracy is a direct value
                    correct += int(losses['accuracy'] * labels.size(0))
                total += labels.size(0)
            else:
                # If logits are not available, skip accuracy calculation for now
                logger.warning("No logits or accuracy found in losses, skipping accuracy calculation")
                continue
            
            # Log training status
            if (batch_idx + 1) % cfg.log_config.interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                logger.info(f'Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Validate if needed
        if not args.no_validate and val_dataloader is not None:
            logger.info('Starting validation...')
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs = inputs.to(cfg.device)
                    labels = labels.to(cfg.device)
                    
                    # Forward pass for validation
                    outputs = model.simple_test(inputs)
                    
                    # Calculate validation accuracy
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_accuracy = 100. * val_correct / val_total
            logger.info(f'Validation Accuracy: {val_accuracy:.2f}%')
            
            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model_path = os.path.join(cfg.work_dir, 'best_model.pth')
                torch.save({'model': model.state_dict(), 'epoch': epoch+1, 'accuracy': val_accuracy}, best_model_path)
                logger.info(f'Saved best model with accuracy: {best_accuracy:.2f}% to {best_model_path}')
            
            # Switch back to training mode
            model.train()
        
        # Save checkpoint at specified intervals
        if (epoch + 1) % cfg.checkpoint_config.interval == 0:
            checkpoint_path = os.path.join(cfg.work_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({'model': model.state_dict(), 'epoch': epoch+1, 'optimizer': optimizer.state_dict()}, checkpoint_path)
            logger.info(f'Saved checkpoint to {checkpoint_path}')
    
    # Save final model
    final_model_path = os.path.join(cfg.work_dir, 'final_model.pth')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, final_model_path)
    logger.info(f'Saved final model to {final_model_path}')
    logger.info(f'Training completed! Best validation accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    main()
