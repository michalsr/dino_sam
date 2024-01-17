import torch
import os
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
import torch
from PIL import Image 
import torchvision.transforms as T
import math
import os
import argparse
import utils
import torch.nn.functional as F
import cv2 
import extract_features as image_features
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

"""
Given extracted regions from SAM, extract image features (if not done already), create feature vectors for each region using some method (eg. avg)
"""

def extract_image_features(args,image_name):
    try:
        cv = cv2.imread(os.path.join(args.image_dir, image_name)+args.image_ext)
        color_coverted = cv2.cvtColor(cv, cv2.COLOR_BGR2RGB) 
        image = Image.fromarray(color_coverted)

    except:
        print(f'Could not read image {image_name}')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device  
    if args.dtype == "fp16":
      args.dtype = torch.half
    elif args.dtype == "fp32":
      args.dtype = torch.float ## this change is needed for CLIP model
    else: 
      args.dtype = torch.bfloat16

    if args.model == 'clip':
        model, preprocess = clip.load(args.clip_model, device=device)
    elif args.model == 'dense_clip':
        model = DenseCLIP('ViT-L/14@336px').to(device)
    elif args.model == 'imagenet':
        model = timm.create_model('vit_large_patch32_224.orig_in21k', pretrained=True, dynamic_img_size = True)
    else:
        model = torch.hub.load(f'{args.model_repo_name}', f'{args.model}')

    model = model.to(device=args.device, dtype=args.dtype)
    try:  
        if 'dino' in args.model:
            if 'dinov2' in args.model:
                features = image_features.extract_dino_v2(args, model, image)
            else:  # dinov1
                features = image_features.extract_dino_v1(args, model, image)

        elif args.model == 'clip':
            features = image_features.extract_clip(args, model, image, preprocess)
        
        elif args.model == 'dense_clip':
            features = image_features.extract_dense_clip(args, model, image)
        
        elif args.model == 'imagenet':
            features = image_features.extract_imagenet(args, model, image)
    except torch.cuda.OutOfMemoryError as e:
            logger.warning(f'Caught CUDA out of memory error for {f}; falling back to CPU')
            torch.cuda.empty_cache()
            features = None 
    return features 

    

def region_features(args,image_id_to_sam):
    if args.feature_dir!= None:
        features_exist = True 
        # Get the intersection of the feature files and the sam regions
        all_feature_files = [f for f in os.listdir(args.feature_dir) if os.path.isfile(os.path.join(args.feature_dir, f))]
        feature_files_in_sam = [f for f in all_feature_files if os.path.splitext(f)[0] in image_id_to_sam]

        features_minus_sam = set(all_feature_files) - set(feature_files_in_sam)
        if len(features_minus_sam) > 0:
            logger.warning(f'Found {len(features_minus_sam)} feature files that are not in the set of SAM region files: {features_minus_sam}')
    else:
        features_exist = False 
        logger.warning('No feature directory. Will extract features while processing features')
    if features_exist:
        prog_bar = tqdm(feature_files_in_sam)

    else:
        prog_bar = tqdm(image_id_to_sam)

    def extract_features(f, args,device='cuda',features_exist=True):
        prog_bar.set_description(f'Region features: {f}')
        if features_exist:
            features = utils.open_file(os.path.join(args.feature_dir,f))
        else:
            # need to extract extract image features 
            features = extract_image_features(args,f)
        file_name = f
        ext = os.path.splitext(f)[1]
        all_region_features_in_image = []
        sam_regions = image_id_to_sam[file_name.replace(ext,'')]

        if args.pooling_method == 'downsample':
            f1, h1, w1 = features[0].shape

            for region in sam_regions:
                sam_region_feature = {}
                sam_region_feature['region_id'] = region['region_id']
                sam_region_feature['area'] = region['area']
                sam_mask = mask_utils.decode(region['segmentation'])
                h2, w2 = sam_mask.shape
                downsampled_mask = torch.from_numpy(sam_mask).cuda()
                downsampled_mask = downsampled_mask.unsqueeze(0).unsqueeze(0)
                downsampled_mask = torch.nn.functional.interpolate(downsampled_mask, size=(h1, w1), mode='nearest').squeeze(0).squeeze(0)

                if torch.sum(downsampled_mask).item() == 0:
                    continue

                features_in_sam = torch.from_numpy(features).cuda().squeeze(dim = 0)[:, downsampled_mask==1].view(f1, -1).mean(1).cpu().numpy()
                sam_region_feature['region_feature'] = features_in_sam
                all_region_features_in_image.append(sam_region_feature)
        else:
            if len(sam_regions) > 0:
                # sam regions within an image all have the same total size
                new_h, new_w = mask_utils.decode(sam_regions[0]['segmentation']).shape
                patch_length = args.dino_patch_length
                padded_h, padded_w = math.ceil(new_h / patch_length) * patch_length, math.ceil(new_w / patch_length) * patch_length # Get the padded height and width
                upsample_feature = torch.nn.functional.interpolate(torch.from_numpy(features).cuda(), size=[padded_h,padded_w],mode='bilinear') # First interpolate to the padded size
                upsample_feature = T.CenterCrop((new_h, new_w)) (upsample_feature).squeeze(dim = 0) # Apply center cropping to the original size
                f,h,w = upsample_feature.size()

                for region in sam_regions:
                    sam_region_feature = {}
                    sam_region_feature['region_id'] = region['region_id']
                    sam_region_feature['area'] = region['area']
                    sam_mask = mask_utils.decode(region['segmentation'])
                    r_1, r_2 = np.where(sam_mask == 1)

                    if args.pooling_method == 'average':
                        features_in_sam = upsample_feature[:,r_1,r_2].view(f,-1).mean(1).cpu().numpy()
                    elif args.pooling_method == 'max':
                        input_max, max_indices = torch.max(upsample_feature[:,r_1,r_2].view(f,-1), 1)
                        features_in_sam = input_max.cpu().numpy()

                    sam_region_feature['region_feature'] = features_in_sam
                    all_region_features_in_image.append(sam_region_feature)
        utils.save_file(os.path.join(args.region_feature_dir, file_name.replace(ext,'.pkl')), all_region_features_in_image)

    for i,f in enumerate(prog_bar):
        try:
            extract_features(f,args,features_exist=features_exist)

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f'Caught CUDA out of memory error for {f}; falling back to CPU')
            torch.cuda.empty_cache()
            extract_features(f,args,features_exist=features_exist, device='cpu')
        # except Exception as e:
        #     print(f'Error: {e}')
        #     continue 

def load_all_regions(args):
    if len(os.listdir(args.mask_dir)) == 0:
        raise Exception(f"No regions found at {args.mask_dir}")
    logger.info(f"Loading region masks from {args.mask_dir}")
    image_id_to_mask = {}
    for f in tqdm(os.listdir(args.mask_dir)):
        filename_extension = os.path.splitext(f)[1]
        regions = utils.open_file(os.path.join(args.mask_dir,f))
        if not args.use_sam:
            regions = [r for r in regions if 'mask' in list(r.keys())]
        image_id_to_mask[f.replace(filename_extension,'')] = regions
    return image_id_to_mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_dir",
        type=str,
        default="/shared/rsaas/dino_sam"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Location of extracted features",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Location of masks (sam or ground truth if given)",
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

    parser.add_argument(
        "--use_sam",
        action="store_false",
        help="If not using json sam regions"
    )
    
    parser.add_argument(
        "--pooling_method",
        type=str,
        default='average',
        choices=['average', 'max', 'downsample'],
        help='pooling methods'
    )

    # extract feature arguments 
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help='Image dir for extracting features'
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp16', 'fp32','bf16'],
        help="Which mixed precision to use. Use fp32 for clip and dense_clip"
    )
    parser.add_argument(
        "--model_repo_name",
        type=str,
        default="facebookresearch/dinov2",
        choices=['facebookresearch/dinov2','facebookresearch/dino:main'],
        help="PyTorch model name for downloading from PyTorch hub"
    )

    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-L/14@336px"],
        help="CLIP base model version"
    )

    parser.add_argument(
        "--model",
        type=str,
        default='dinov2_vitl14',
        choices=['dinov2_vitl14', 'dino_vitb8', 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitg14', 'clip','dino_vitb16','dense_clip', 'imagenet'],  
        help="Name of model from repo"
    )

    parser.add_argument(
        "--layers",
        type=str,
        default="[23]",
        help="List of layers or number of last layers to take"
    )
    parser.add_argument(
        "--padding",
        default="center",
        help="Padding used for transforms"
    )
    parser.add_argument(
        "--image_ext",
        default=".jpg",
        help="Image extension for reading from dir"
    )
    parser.add_argument(
        "--multiple",
        type=int,
        default=14,
        help="The patch length of the model. Use 14 for DINOv2, 8 for DINOv1, 32 for CLIP, 14 for DenseCLIP (automatically handled in the package)"
    )

    args = parser.parse_args()


    image_id_to_mask = load_all_regions(args)
    region_features(args,image_id_to_mask)

    logger.info('Done')

