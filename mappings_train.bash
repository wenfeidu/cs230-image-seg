#!/bin/bash  

for filename in $SCRATCH/raw_data/*.txt; do
    direname=$(basename "$filename" .txt)
    mkdir $SCRATCH/raw_data/train_color/$direname
    mkdir $SCRATCH/raw_data/train_label/$direname
    cut -f 1 $filename | cat | while read -r line
    do
        jpg_file=${line//\\//}
        jpg_file_base=$(basename "$jpg_file")
        jpg_file_path="$SCRATCH/raw_data/train_color/$jpg_file_base"
        mv $jpg_file_path "$SCRATCH/raw_data/train_color/$direname"
    done
    cut -f 2 $filename | cat | while read -r line
    do
        png_file=${line//\\//}
        png_file_base=$(basename "$png_file")
        png_file_path="$SCRATCH/raw_data/train_label/$png_file_base"   
        mv $png_file_path "$SCRATCH/raw_data/train_label/$direname"   
    done
done
