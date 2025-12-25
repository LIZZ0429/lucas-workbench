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

def copy_files(src_dir, dst_dir, max_samples=None):
    """
    Copies files from src_dir to dst_dir, with optional max_samples for sampling
    Returns number of copied files
    """
    create_directory(dst_dir)
    
    # Get all files in source directory
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    if not files:
        logger.warning(f"No files found in {src_dir}")
        return 0
    
    # Sample files if max_samples is specified and less than total files
    if max_samples and len(files) > max_samples:
        sampled_files = random.sample(files, max_samples)
        logger.info(f"Sampled {max_samples} files from {len(files)} total files in {src_dir}")
    else:
        sampled_files = files
        logger.info(f"Copying all {len(files)} files from {src_dir}")
    
    # Copy files
    copied_count = 0
    for file in sampled_files:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(dst_dir, file)
        try:
            shutil.copy2(src_file, dst_file)
            copied_count += 1
        except Exception as e:
            logger.error(f"Error copying {src_file} to {dst_file}: {e}")
    
    logger.info(f"Successfully copied {copied_count} files from {src_dir} to {dst_dir}")
    return copied_count

def migrate_class_from_sampled(src_base, dst_base, class_name, is_incremental=False):
    """
    Migrates a class from CICIDS2017-sampled directory
    """
    logger.info(f"\n=== Migrating class: {class_name} ===")
    
    # Create class directory
    class_dir = os.path.join(dst_base, class_name)
    create_directory(class_dir)
    
    # Migrate train split
    train_src = os.path.join(src_base, "train", class_name)
    train_dst = os.path.join(class_dir, "train")
    if os.path.exists(train_src):
        max_samples = 10 if is_incremental else None
        copy_files(train_src, train_dst, max_samples)
    else:
        logger.warning(f"Train directory not found for {class_name}: {train_src}")
    
    # Migrate valid split - copy all files
    valid_src = os.path.join(src_base, "valid", class_name)
    valid_dst = os.path.join(class_dir, "valid")
    if os.path.exists(valid_src):
        copy_files(valid_src, valid_dst)
    else:
        logger.warning(f"Valid directory not found for {class_name}: {valid_src}")
    
    # Migrate test split - copy all files
    test_src = os.path.join(src_base, "test", class_name)
    test_dst = os.path.join(class_dir, "test")
    if os.path.exists(test_src):
        copy_files(test_src, test_dst)
    else:
        logger.warning(f"Test directory not found for {class_name}: {test_src}")

def migrate_class_from_unknown(src_dir, dst_base, class_name):
    """
    Migrates a class from CICIDS2017-UNKOWN-PNG directory
    This directory doesn't have train/valid/test splits, so we need to handle it differently
    """
    logger.info(f"\n=== Migrating UNKNOWN class: {class_name} ===")
    
    # Create class directory and split subdirectories
    class_dir = os.path.join(dst_base, class_name)
    create_directory(class_dir)
    
    train_dst = os.path.join(class_dir, "train")
    valid_dst = os.path.join(class_dir, "valid")
    test_dst = os.path.join(class_dir, "test")
    
    # Get all files in source directory
    files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    if not files:
        logger.warning(f"No files found in {src_dir}")
        return
    
    logger.info(f"Processing {len(files)} files for {class_name}")
    
    # For UNKNOWN classes, we need to:
    # 1. Copy 10 random files to train
    # 2. Copy all files to valid and test
    
    # Sample 10 files for train
    train_files = random.sample(files, 10) if len(files) > 10 else files
    logger.info(f"Selected {len(train_files)} files for training from {class_name}")
    
    # Copy train files
    create_directory(train_dst)
    copied_train = 0
    for file in train_files:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(train_dst, file)
        try:
            shutil.copy2(src_file, dst_file)
            copied_train += 1
        except Exception as e:
            logger.error(f"Error copying {src_file} to {dst_file}: {e}")
    logger.info(f"Successfully copied {copied_train} files to train directory for {class_name}")
    
    # Copy all files to valid
    logger.info(f"Copying all files to valid directory for {class_name}")
    copy_files(src_dir, valid_dst)
    
    # Copy all files to test
    logger.info(f"Copying all files to test directory for {class_name}")
    copy_files(src_dir, test_dst)

def main():
    """
    Main function to perform data migration according to user requirements
    """
    # Source and destination paths
    src_sampled = "/root/autodl-tmp/NetMamba/data/CICIDS2017-sampled"
    src_unknown = "/root/autodl-tmp/NetMamba/data/CICIDS2017-UNKOWN-PNG"
    dst_base = "/root/autodl-tmp/NewModel/data/CICIDS2017"
    
    logger.info("Starting data migration...")
    logger.info(f"Source (sampled): {src_sampled}")
    logger.info(f"Source (unknown): {src_unknown}")
    logger.info(f"Destination: {dst_base}")
    
    # Create destination base directory
    create_directory(dst_base)
    
    # 1. Base classes migration - copy all files
    # Note: Benign is already migrated, no need to re-migrate
    base_classes = ["DDoS-LOIT", "Dos-GoldenEye", "Botnet"]
    logger.info("\n=== Migrating Base Classes (all files) ===")
    for cls in base_classes:
        migrate_class_from_sampled(src_sampled, dst_base, cls, is_incremental=False)
    
    # 2. First incremental round - Dos-Hulk, FTP-Brute-Force
    # train: 10 samples, test/valid: all files
    incremental_round1 = ["Dos-Hulk", "FTP-Brute-Force"]
    logger.info("\n=== Migrating First Incremental Round (10 samples for train) ===")
    for cls in incremental_round1:
        migrate_class_from_sampled(src_sampled, dst_base, cls, is_incremental=True)
    
    # 3. Migrate UNKNOWN classes from CICIDS2017-UNKOWN-PNG
    # These classes don't have train/valid/test splits
    # train: 10 samples, test/valid: all files
    logger.info("\n=== Migrating UNKNOWN Classes (10 samples for train) ===")
    unknown_classes = [
        "Dos-Slowhttptest", 
        "SSH-Brute-Force", 
        "Web-Attack-Brute-Force", 
        "Web-Attack-Sql-Injection", 
        "Web-Attack-XSS"
    ]
    
    for cls in unknown_classes:
        src_dir = os.path.join(src_unknown, cls)
        if os.path.exists(src_dir):
            migrate_class_from_unknown(src_dir, dst_base, cls)
        else:
            logger.warning(f"Directory not found for unknown class {cls}: {src_dir}")
    
    logger.info("\n=== Data Migration Completed ===")
    logger.info(f"All migration tasks have been completed successfully!")

if __name__ == "__main__":
    main()