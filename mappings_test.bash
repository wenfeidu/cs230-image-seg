#!/bin/bash

for filename in $SCRATCH/raw_data/list_test_mapping/*.txt; do
    cat $filename | while read -r line
    do
        files=${line//\\//}
        old_file_base=$(cut -f 1 -d ' ' <<< $files)
        new_file=$(cut -f 2 -d $'\t' <<< $files)
        old_file_path="$SCRATCH/raw_data/test/$old_file_base.jpg"
        new_file_base=$(basename "$new_file")
        new_file_path="$SCRATCH/raw_data/test/$new_file_base"
        echo $new_file_path
        echo $old_file_path
        mv $old_file_path $new_file_path
    done
done
for filename in $SCRATCH/raw_data/list_test/*.txt; do
    direname=$(basename "$filename" .txt)
    mkdir $SCRATCH/raw_data/test/$direname
    cat $filename | while read -r line
    do
        jpg_file=${line//\\//}
        jpg_file_base=$(basename "$jpg_file")
        jpg_file_path="$SCRATCH/raw_data/test/$jpg_file_base"
        mv $jpg_file_path "$SCRATCH/raw_data/test/$direname"
    done
done
