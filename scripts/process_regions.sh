#!/bin/bash
python process_regions.py  --feature_dir /shared/rsaas/dino_sam/features/roxford/query --mask_dir /shared/rsaas/dino_sam/sam_output/roxford/query --region_feature_dir /shared/rsaas/dino_sam/region_features/dinov2/roxford/query 
chmod 777 -R /shared/rsaas/dino_sam/region_features/dinov2/roxford/query 