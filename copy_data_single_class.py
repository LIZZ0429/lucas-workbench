import os
import shutil
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directory(path):
    """
    Create directory if it doesn't exist
    """
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")

def copy_directory(src, dst, max_samples=None):
    """
    Copies files from src to dst, with optional max_samples for sampling
    Returns number of copied files
    """
    create_directory(dst)
    
    # Get all files in source directory
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
    
    # Sample files if max_samples is specified and less than total files
    if max_samples and len(files) > max_samples:
        sampled_files = random.sample(files, max_samples)
        logger.info(f"Sampled {max_samples} files from {len(files)} total files in {src}")
    else:
        sampled_files = files
        logger.info(f"Copying all {len(files)} files from {src}")
    
    # Copy files
    copied_count = 0
    for file in sampled_files:
        src_file = os.path.join(src, file)
        dst_file = os.path.join(dst, file)
        try:
            shutil.copy2(src_file, dst_file)
            copied_count += 1
        except Exception as e:
            logger.error(f"Error copying {src_file} to {dst_file}: {e}")
    
    logger.info(f"Successfully copied {copied_count} files from {src} to {dst}")
    return copied_count

def main():
    """
    Single class validation script
    Tests copying a single class with the correct directory structure
    """
    # Source and destination paths
    src_base = "/root/autodl-tmp/NetMamba/data/CICIDS2017-sampled"
    dst_base = "/root/autodl-tmp/NewModel/data"
    
    # Test class
    test_class = "Benign"
    
    logger.info(f"Starting single class validation for {test_class}...")
    logger.info(f"Source: {src_base}")
    logger.info(f"Destination: {dst_base}")
    
    # Create CICIDS2017 directory structure
    cicids2017_dir = os.path.join(dst_base, "CICIDS2017")
    create_directory(cicids2017_dir)
    
    # Create class directory and split subdirectories
    class_dir = os.path.join(cicids2017_dir, test_class)
    create_directory(class_dir)
    
    # Copy all splits for this class
    for split in ["train", "valid", "test"]:
        src_dir = os.path.join(src_base, split, test_class)
        if os.path.exists(src_dir):
            dst_dir = os.path.join(class_dir, split)
            copy_directory(src_dir, dst_dir)
        else:
            logger.warning(f"Directory {src_dir} does not exist")
    
    # Test incremental sampling on a small class
    logger.info("\n=== Testing incremental sampling (10 files) ===")
    sample_class = "Dos-Slowloris"
    sample_dir = os.path.join(cicids2017_dir, sample_class)
    create_directory(sample_dir)
    
    for split in ["train", "valid", "test"]:
        src_dir = os.path.join(src_base, split, sample_class)
        if os.path.exists(src_dir):
            dst_dir = os.path.join(sample_dir, split)
            copy_directory(src_dir, dst_dir, max_samples=10)
        else:
            logger.warning(f"Directory {src_dir} does not exist")
    
    logger.info(f"\n=== Single Class Validation Completed ===")
    logger.info(f"Tested copying all files for class: {test_class}")
    logger.info(f"Tested sampling 10 files for class: {sample_class}")
    logger.info(f"Output directory structure:")
    
    # Show directory structure
    for root, dirs, files in os.walk(dst_base):
        level = root.replace(dst_base, '').count(os.sep)
        indent = ' ' * 2 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files only
            logger.info(f"{subindent}{file}")
        if len(files) > 3:
            logger.info(f"{subindent}... and {len(files) - 3} more files")

if __name__ == "__main__":
    main()