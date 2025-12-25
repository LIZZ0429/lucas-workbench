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
    # Source and destination paths
    src_base = "/root/autodl-tmp/NetMamba/data/CICIDS2017-sampled"
    dst_base = "/root/autodl-tmp/NewModel/data"
    
    # Define migration plan
    # Base classes: copy all files
    base_classes = ["Benign", "DDoS-LOIT", "Dos-GoldenEye", "Dos-Hulk"]
    
    # Incremental classes: 2 classes per round, 10 samples per class
    incremental_rounds = [
        ["Dos-Slowloris", "FTP-Brute-Force"],
        ["Z0-Botnet", "Z1-SSH-Brute-Force"],
        ["Z2-Dos-Slowhttptest", "Z3-Web-Attack-Brute-Force"],
        ["Z4-Web-Attack-Sql-Injection", "Z5-Web-Attack-XSS"]
    ]
    
    logger.info("Starting data migration...")
    logger.info(f"Source: {src_base}")
    logger.info(f"Destination: {dst_base}")
    
    # Create CICIDS2017 directory structure
    cicids2017_dir = os.path.join(dst_base, "CICIDS2017")
    create_directory(cicids2017_dir)
    
    # First, let's do a single-class validation for the Benign class
    logger.info("\n=== Single Class Validation (Benign) ===")
    validation_class = "Benign"
    
    # Create class directory and split subdirectories
    class_dir = os.path.join(cicids2017_dir, validation_class)
    create_directory(class_dir)
    
    for split in ["train", "valid", "test"]:
        src_dir = os.path.join(src_base, split, validation_class)
        dst_dir = os.path.join(class_dir, split)
        copy_directory(src_dir, dst_dir)
    
    logger.info("\n=== Single Class Validation Completed ===")
    
    # Wait for user confirmation before proceeding
    input("Press Enter to continue with the full migration...")
    
    # Copy remaining base classes (all files)
    logger.info("\n=== Copying Base Classes (all files) ===")
    for cls in base_classes[1:]:  # Skip Benign as it's already copied
        # Create class directory
        class_dir = os.path.join(cicids2017_dir, cls)
        create_directory(class_dir)
        
        for split in ["train", "valid", "test"]:
            src_dir = os.path.join(src_base, split, cls)
            if os.path.exists(src_dir):  # Check if class exists in split
                dst_dir = os.path.join(class_dir, split)
                copy_directory(src_dir, dst_dir)
    
    # Process incremental rounds
    for round_num, inc_classes in enumerate(incremental_rounds, 1):
        logger.info(f"\n=== Processing Incremental Round {round_num}: {inc_classes} ===")
        
        for cls in inc_classes:
            # Create class directory
            class_dir = os.path.join(cicids2017_dir, cls)
            create_directory(class_dir)
            
            for split in ["train", "valid", "test"]:
                src_dir = os.path.join(src_base, split, cls)
                if os.path.exists(src_dir):  # Check if class exists in split
                    dst_dir = os.path.join(class_dir, split)
                    copy_directory(src_dir, dst_dir, max_samples=10)
    
    logger.info("\n=== Data Migration Completed ===")
    logger.info(f"Migration plan executed successfully!")
    logger.info(f"Base classes: {base_classes}")
    logger.info(f"Incremental rounds: {incremental_rounds}")

if __name__ == "__main__":
    main()