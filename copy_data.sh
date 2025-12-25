#!/bin/bash

# 源数据位置
SRC_SAMPLED="/root/autodl-tmp/NetMamba/data/CICIDS2017-sampled"
SRC_UNKNOWN="/root/autodl-tmp/NetMamba/data/CICIDS2017-UNKOWN-PNG"

# 目标数据位置
DEST="/root/autodl-tmp/NewModel/data/CICIDS2017"

# 创建目标目录
mkdir -p $DEST

# 日志文件
LOG_FILE="/root/autodl-tmp/NewModel/copy_data.log"
echo "Starting data copy at $(date)" > $LOG_FILE

# 处理CICIDS2017-sampled中的6个类别
echo "Processing CICIDS2017-sampled classes..." >> $LOG_FILE
sampled_classes=("Benign" "DDoS-LOIT" "Dos-GoldenEye" "Dos-Hulk" "Dos-Slowloris" "FTP-Brute-Force")

for class in "${sampled_classes[@]}"; do
    echo "Processing class: $class" >> $LOG_FILE
    
    # 创建目标目录
    mkdir -p $DEST/$class/train $DEST/$class/val $DEST/$class/test
    
    # 复制数据
    echo "Copying train data for $class..." >> $LOG_FILE
    cp -r $SRC_SAMPLED/train/$class/* $DEST/$class/train/ 2>/dev/null || echo "No train data for $class" >> $LOG_FILE
    
    echo "Copying valid data for $class..." >> $LOG_FILE
    cp -r $SRC_SAMPLED/valid/$class/* $DEST/$class/val/ 2>/dev/null || echo "No valid data for $class" >> $LOG_FILE
    
    echo "Copying test data for $class..." >> $LOG_FILE
    cp -r $SRC_SAMPLED/test/$class/* $DEST/$class/test/ 2>/dev/null || echo "No test data for $class" >> $LOG_FILE
    
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
    mkdir -p $DEST/$class/train $DEST/$class/val $DEST/$class/test
    
    # 复制所有文件到临时目录
    temp_dir="/tmp/$class"
    mkdir -p $temp_dir
    
    echo "Copying all files for $class to temp directory..." >> $LOG_FILE
    cp -r $SRC_UNKNOWN/$class/* $temp_dir/ 2>/dev/null || echo "No files for $class" >> $LOG_FILE
    
    # 获取所有PNG文件
    files=($temp_dir/*.png)
    total=${#files[@]}
    
    echo "Total files for $class: $total" >> $LOG_FILE
    
    if [ $total -eq 0 ]; then
        echo "No files found for $class, skipping..." >> $LOG_FILE
        rm -rf $temp_dir
        continue
    fi
    
    # 计算划分数量
    train_num=$((total * 8 / 10))
    val_num=$((total * 1 / 10))
    test_num=$((total - train_num - val_num))
    
    echo "Train: $train_num, Val: $val_num, Test: $test_num" >> $LOG_FILE
    
    # 随机打乱文件列表
    shuffled=($(shuf -e "${files[@]}"))
    
    # 分配文件
    for ((i=0; i<train_num; i++)); do
        cp ${shuffled[$i]} $DEST/$class/train/ 2>/dev/null
    done
    
    for ((i=train_num; i<train_num+val_num; i++)); do
        cp ${shuffled[$i]} $DEST/$class/val/ 2>/dev/null
    done
    
    for ((i=train_num+val_num; i<total; i++)); do
        cp ${shuffled[$i]} $DEST/$class/test/ 2>/dev/null
    done
    
    # 统计文件数量
    train_count=$(ls -1 $DEST/$class/train/ 2>/dev/null | wc -l)
    val_count=$(ls -1 $DEST/$class/val/ 2>/dev/null | wc -l)
    test_count=$(ls -1 $DEST/$class/test/ 2>/dev/null | wc -l)
    
    echo "Class $class: Train=$train_count, Val=$val_count, Test=$test_count" >> $LOG_FILE
    
    # 清理临时目录
    rm -rf $temp_dir
done

echo "\nData copy completed at $(date)" >> $LOG_FILE
echo "Data copy completed!"