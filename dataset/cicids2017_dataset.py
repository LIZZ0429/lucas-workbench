import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import glob
import logging

# 设置日志记录器
logger = logging.getLogger(__name__)

class CICIDS2017Dataset(Dataset):
    """
    CICIDS2017数据集加载器，用于加载转换后的PNG图像。
    支持增量学习的任务设置，包括base classes和novel classes。
    """
    def __init__(self, data_prefix, pipeline=None, split='train', num_classes=15, base_classes=5, 
                 task_id=0, num_tasks=3, test_mode=False, **kwargs):
        """
        初始化数据集
        
        Args:
            data_prefix (str): Path to the dataset root directory
            pipeline (list, optional): Data augmentation pipeline
            split (str): Dataset split, one of ['train', 'val', 'test']
            num_classes (int): Total number of classes in the dataset
            base_classes (int): Number of base classes
            task_id (int): Current task ID (0-based)
            num_tasks (int): Total number of tasks
            test_mode (bool): Whether to use test mode
        """
        self.data_prefix = data_prefix
        self.pipeline = pipeline
        self.split = split
        self.num_classes = num_classes
        self.base_classes = base_classes
        self.task_id = task_id
        self.num_tasks = num_tasks
        self.test_mode = test_mode
        
        # 获取所有类别的名称
        self.class_names = sorted(os.listdir(data_prefix))
        self.actual_num_classes = len(self.class_names)
        
        # 建立类别到索引的映射
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # 确定当前任务的类别范围
        self._setup_task_classes()
        
        # 加载数据样本
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for task {task_id} ({self.start_class}-{self.end_class}) - Split: {split}")
    
    def _setup_task_classes(self):
        """
        设置当前任务的类别范围
        """
        if self.test_mode and self.split == 'test':
            # 只有测试集在test_mode下使用所有类别
            self.start_class = 0
            self.end_class = self.actual_num_classes
        elif self.task_id == 0:
            # 第一个任务处理基础类别
            self.start_class = 0
            self.end_class = self.base_classes
        else:
            # 后续任务处理增量类别
            remaining_classes = self.actual_num_classes - self.base_classes
            classes_per_task = remaining_classes // (self.num_tasks - 1)
            
            self.start_class = self.base_classes + (self.task_id - 1) * classes_per_task
            if self.task_id == self.num_tasks - 1:
                # 最后一个任务处理剩余的所有类别
                self.end_class = self.actual_num_classes
            else:
                self.end_class = self.base_classes + self.task_id * classes_per_task
        
        # 对于验证集，确保使用与训练集相同的类别范围
        # 只有在test_mode下的测试集才使用所有类别
        if self.split == 'val' and not (self.test_mode and self.split == 'test'):
            # 验证集使用与训练集相同的类别范围
            if self.task_id == 0:
                self.start_class = 0
                self.end_class = self.base_classes
            else:
                remaining_classes = self.actual_num_classes - self.base_classes
                classes_per_task = remaining_classes // (self.num_tasks - 1)
                
                self.start_class = self.base_classes + (self.task_id - 1) * classes_per_task
                if self.task_id == self.num_tasks - 1:
                    self.end_class = self.actual_num_classes
                else:
                    self.end_class = self.base_classes + self.task_id * classes_per_task
    
    def _load_samples(self):
        """
        加载当前任务的所有样本
        """
        samples = []
        
        # 遍历当前任务的所有类别
        for class_idx in range(self.start_class, self.end_class):
            class_name = self.class_names[class_idx]
            
            # 根据split参数构建不同的目录路径
            if self.split == 'val':
                # 处理val和valid的情况
                split_dir = 'valid'
            else:
                split_dir = self.split
            
            class_dir = os.path.join(self.data_prefix, class_name, split_dir)
            
            if not os.path.isdir(class_dir):
                logger.warning(f"Directory {class_dir} not found, skipping...")
                continue
            
            # 获取所有PNG图像文件
            image_files = glob.glob(os.path.join(class_dir, "*.png"))
            
            # 为每个图像创建样本
            for img_path in image_files:
                samples.append((img_path, class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        """
        img_path, label = self.samples[idx]
        
        # 读取图像
        img = Image.open(img_path).convert('L')  # 转换为灰度图
        
        # 应用数据增强
        if self.pipeline is not None:
            # TODO: Implement pipeline processing
            pass
        
        # 转换为张量
        if isinstance(img, Image.Image):
            img = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0)  # 添加通道维度
        
        return img, label
    
    @property
    def CLASSES(self):
        """
        获取所有类别的名称
        """
        return self.class_names
    
    def get_class_names(self):
        """
        获取当前任务的类别名称
        """
        return self.class_names[self.start_class:self.end_class]
    
    def get_num_classes(self):
        """
        获取当前任务的类别数量
        """
        return self.end_class - self.start_class


def build_dataset(cfg, default_args=None):
    """
    Build a dataset from config dict.
    
    Args:
        cfg (dict): Config dict.
        default_args (dict, optional): Default arguments.
        
    Returns:
        Dataset: The constructed dataset.
    """
    if default_args is None:
        default_args = {}
    
    dataset_type = cfg.pop('type')
    
    if dataset_type == 'CICIDS2017Dataset':
        return CICIDS2017Dataset(**cfg, **default_args)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def build_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
    """
    Build a dataloader from dataset.
    
    Args:
        dataset (Dataset): Dataset instance.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of workers.
        pin_memory (bool): Whether to use pin memory.
        
    Returns:
        DataLoader: The constructed dataloader.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
