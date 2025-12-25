#!/bin/bash

# 源数据位置
SRC_SAMPLED="/root/autodl-tmp/NetMamba/data/CICIDS2017-sampled"
SRC_UNKNOWN="/root/autodl-tmp/NetMamba/data/CICIDS2017-UNKOWN-PNG"

# 目标数据位置
DEST="/root/autodl-tmp/NewModel/data/CICIDS2017"

# 创建目标目录
mkdir -p $DEST

# 日志文件
LOG_FILE="/root/autodl-tmp/NewModel/copy_data_symlink.log"
echo "Starting data symlink at $(date)" > $LOG_FILE

# 处理CICIDS2017-sampled中的6个类别
echo "Processing CICIDS2017-sampled classes..." >> $LOG_FILE
sampled_classes=("Benign" "DDoS-LOIT" "Dos-GoldenEye" "Dos-Hulk" "Dos-Slowloris" "FTP-Brute-Force")

for class in "${sampled_classes[@]}"; do
    echo "Processing class: $class" >> $LOG_FILE
    
    # 创建目标目录
    mkdir -p $DEST/$class
    
    # 创建符号链接
    echo "Creating symlinks for $class..." >> $LOG_FILE
    ln -sf $SRC_SAMPLED/train/$class $DEST/$class/train
    ln -sf $SRC_SAMPLED/valid/$class $DEST/$class/val
    ln -sf $SRC_SAMPLED/test/$class $DEST/$class/test
    
    # 统计文件数量
    train_count=$(ls -1 $DEST/$class/train/ 2>/dev/null | wc -l)
    val_count=$(ls -1 $DEST/$class/val/ 2>/dev/null | wc -l)
    test_count=$(ls -1 $DEST/$class/test/ 2>/dev/null | wc -l)
    
    echo "Class $class: Train=$train_count, Val=$val_count, Test=$test_count" >> $LOG_FILE
done

# 处理CICIDS2017-UNKOWN-PNG中的6个类别
echo "\nProcessing CICIDS2017-UNKOWN-PNG classes..." >> $LOG_FILE
unknown_classes=("Botnet" "Dos-Slowhttptest" "SSH-Brute-Force" "Web-Attack-Brute-Force" "Web-Attack-Sql-Injection" "Web-Attack-XSS")

for class in "${unknown_classes[@]}"; do
    echo "Processing class: $class" >> $LOG_FILE
    
    # 创建目标目录
    mkdir -p $DEST/$class
    
    # 由于未知类别没有train/val/test划分，我们将其直接链接
    # 数据集加载器会根据split参数选择不同的划分策略
    echo "Creating symlink for $class..." >> $LOG_FILE
    ln -sf $SRC_UNKNOWN/$class $DEST/$class/train
    ln -sf $SRC_UNKNOWN/$class $DEST/$class/val
    ln -sf $SRC_UNKNOWN/$class $DEST/$class/test
    
    # 统计文件数量
    total_count=$(ls -1 $SRC_UNKNOWN/$class/ 2>/dev/null | wc -l)
    echo "Class $class: Total=$total_count" >> $LOG_FILE
done

echo "\nData symlink completed at $(date)" >> $LOG_FILE
echo "Data symlink completed!"