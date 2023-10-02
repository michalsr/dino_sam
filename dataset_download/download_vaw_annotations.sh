#!/bin/bash
#vaw annotations. images for vaw are from visual genome
filePath=$1
mkdir -p $filePath 
cd $filePath
if [ ! -f "$filePath/train_part1.json" ]; then  
    wget https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/train_part1.json
fi 
if [ ! -f "$filePath/train_part2.json" ]; then 
    wget https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/train_part2.json
fi 
if [ ! -f "$filePath/attribute_index.json" ]; then 
    wget https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/attribute_index.json
fi 
if [ ! -f "$filePath/attribute_parent_types.json" ]; then 
    wget https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/attribute_parent_types.json
fi 
if [ ! -f "$filePath/head_tail.json" ]; then
    wget https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/head_tail.json
fi 
if [ ! -f "$filePath/test.json" ]; then
    wget https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/test.json
fi 
if [ ! -f "$filePath/val.json" ]; then
    wget https://raw.githubusercontent.com/adobe-research/vaw_dataset/main/data/val.json
fi 