#!/bin/bash
filePath=$1/visual_genome
mkdir -p $filePath
cd $filePath
if [ ! -d "$filePath/images" ]; then 
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
    unzip images.zip 
    wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
    unzip images2.zip 
    imagePath="$filePath/images"
    mkdir -p $imagePath
    mv VG_100K_2/* $imagePath
    mv VG_100K/* $imagePath 
    rm -r VG_100K_2
    rm -r VG_100K
fi 
