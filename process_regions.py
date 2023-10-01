import torch 
import pickle
import os 
import numpy as np
import json 
from tqdm import tqdm
from pycocotools import mask as mask_utils
import torch  
from PIL import Image 
import torchvision.transforms as T
import itertools
import math 
import os
import argparse
import utils 
import torch.nn.functional as F

"""
Given extracted regions from SAM and image features, create feature vectors for each region using some method (eg. avg)
"""

def region_features(args,image_id_to_sam):
    all_feature_files = [f for f in os.listdir(args.feature_dir) if os.path.isfile(os.path.join(args.feature_dir, f))]
    for i,f in enumerate(tqdm(all_feature_files,desc='Region features',total=len(all_feature_files))):
        features = utils.open_file(os.path.join(args.feature_dir,f))
        file_name =f 
        ext = os.path.splitext(f)[1]  
        all_region_features_in_image = []
        sam_regions = image_id_to_sam[file_name.replace(ext,'')]
        # sam regions within an image all have the same total size 
        new_h, new_w = mask_utils.decode(sam_regions[0]['segmentation']).shape
        patch_length = args.dino_patch_length
        padded_h, padded_w = math.ceil(new_h / patch_length) * patch_length, math.ceil(new_w / patch_length) * patch_length # Get the padded height and width
        upsample_feature = torch.nn.functional.upsample(features,size=[padded_h,padded_w],mode='bilinear') # First interpolate to the padded size
        upsample_feature = T.CenterCrop((new_h, new_w), upsample_feature).squeeze(dim = 0) # Apply center cropping to the original size
        f,h,w = upsample_feature.size()
        for region in sam_regions:
                sam_region_feature = {}
                sam_region_feature['region_id'] = region['region_id']
                sam_region_feature['area'] = region['area']
                sam_mask = mask_utils.decode(region['segmentation'])
                r_1, r_2 = np.where(sam_mask == 1)
                features_in_sam = upsample_feature[:,r_1,r_2].view(f,-1).mean(1).cpu().numpy()
                sam_region_feature['region_feature'] = features_in_sam
                all_region_features_in_image.append(sam_region_feature)
        utils.save_file(os.path.join(args.region_feature_dir,file_name.replace(ext,'.pkl')),all_region_features_in_image)

def load_all_sam_regions(args):
    if len(os.listdir(args.sam_dir)) == 0:
        raise Exception(f"No sam regions found at {args.sam_dir}")
    print(f"Loading sam regions from {args.sam_dir}")
    image_id_to_sam = {}
    for f in tqdm(os.listdir(args.sam_dir)):
        sam_regions = utils.open_file(os.path.join(args.sam_dir,f))
        image_id_to_sam[f.replace('.json','')] = sam_regions
    return image_id_to_sam

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_dir",
        type=str,
        default="/shared/rsaas/dino_sam"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Location of jpg files",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Location of extracted features",
    )
    parser.add_argument(
        "--sam_dir",
        type=str,
        default=None,
        help="Location of sam masks",
    )

    parser.add_argument(
        "--region_feature_dir",
        type=str,
        default=None,
        help="Location of features per region/pooled features",
    )

    parser.add_argument(
        "--dino_patch_length",
        type=int,
        default=14,
        help="the length of dino patch",
    )

    args = parser.parse_args()
    image_id_to_sam = load_all_sam_regions(args)
    region_features(args,image_id_to_sam)
