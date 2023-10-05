#!/bin/bash
source /home/michal5/anaconda3/bin/activate dino_sam
cd /home/michal5/dino_sam
#sh dataset_download/download_visual_genome.sh /data/shared/visual_genome 
python extract_features.py --dtype bf16 --image_dir /scratch/michal5/vg/images --feature_dir /shared/rsaas/dino_sam/features/vaw --layers [23]