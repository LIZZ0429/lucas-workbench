import torch
import torch.nn.functional as F
import numpy as np

class MemoryModule:
    """
    Memory Module for incremental learning.
    Stores class centers for each seen class.
    
    Args:
        feature_dim (int): Dimension of the feature vectors.
        device (str): Device to use for storage.
    """
    def __init__(self, feature_dim, device='cuda'):
        self.feature_dim = feature_dim
        self.device = device
        self.memory = {}
        self.num_classes = 0
    
    def update(self, class_id, features):
        """
        Update the memory with features for a specific class.
        
        Args:
            class_id (int): Class identifier.
            features (Tensor): Features for the class, shape [N, feature_dim].
        """
        # Compute the mean of the features
        class_center = features.mean(dim=0)
        
        # L2 normalization
        class_center = F.normalize(class_center, dim=0)
        
        # Update memory
        self.memory[class_id] = class_center.to(self.device)
        self.num_classes = len(self.memory)
    
    def get_class_center(self, class_id):
        """
        Get the class center for a specific class.
        
        Args:
            class_id (int): Class identifier.
        
        Returns:
            Tensor: Class center feature, shape [feature_dim].
        """
        assert class_id in self.memory, f"Class {class_id} not in memory"
        return self.memory[class_id]
    
    def get_all_class_centers(self):
        """
        Get all class centers.
        
        Returns:
            Tensor: All class centers, shape [num_classes, feature_dim].
        """
        if self.num_classes == 0:
            return None
        
        # Sort classes by id
        sorted_classes = sorted(self.memory.keys())
        class_centers = torch.stack([self.memory[class_id] for class_id in sorted_classes], dim=0)
        return class_centers
    
    def get_num_classes(self):
        """
        Get the number of classes in memory.
        
        Returns:
            int: Number of classes.
        """
        return self.num_classes
    
    def reset(self):
        """
        Reset the memory.
        """
        self.memory = {}
        self.num_classes = 0
    
    def save(self, path):
        """
        Save the memory to a file.
        
        Args:
            path (str): Path to save the memory.
        """
        memory_dict = {
            'feature_dim': self.feature_dim,
            'memory': {k: v.cpu().numpy() for k, v in self.memory.items()}
        }
        torch.save(memory_dict, path)
    
    def load(self, path):
        """
        Load the memory from a file.
        
        Args:
            path (str): Path to load the memory from.
        """
        memory_dict = torch.load(path)
        self.feature_dim = memory_dict['feature_dim']
        self.memory = {k: torch.tensor(v, device=self.device) for k, v in memory_dict['memory'].items()}
        self.num_classes = len(self.memory)
    
    def compute_class_centers(self, model, dataloader):
        """
        Compute class centers using a model and dataloader.
        
        Args:
            model (nn.Module): Model to extract features.
            dataloader (DataLoader): Dataloader to iterate over data.
        """
        model.eval()
        features_dict = {}
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Extract features from the model's projector output
                features = model.extract_feat(inputs, stage='neck')
                
                # Get the actual feature tensor from the dictionary
                if isinstance(features, dict):
                    features = features['out']
                
                # Process each sample
                for feature, label in zip(features, labels):
                    label = label.item()
                    if label not in features_dict:
                        features_dict[label] = []
                    features_dict[label].append(feature)
        
        # Compute class centers
        for class_id, features_list in features_dict.items():
            features = torch.stack(features_list, dim=0)
            self.update(class_id, features)
    
    def __repr__(self):
        return f"MemoryModule(feature_dim={self.feature_dim}, num_classes={self.num_classes}, device={self.device})"
