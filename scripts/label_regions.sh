#!/bin/bash
python label_regions.py --region_labels /shared/rsaas/dino_sam/region_labels/ADE20K/val --annotation_dir /shared/rsaas/dino_sam/data/ADE20K/annotations/validation --num_classes 150 --sam_dir /shared/rsaas/dino_sam/sam_output/ADE20K/val