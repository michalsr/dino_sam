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
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple = 14):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output
def extract_dino_v2(model,image):

    
    model = model.to(device='cuda',dtype=torch.float32)
    transform = T.Compose([
        T.ToTensor(),
        lambda x: x.unsqueeze(0),

        CenterPadding(multiple = 14),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    with torch.inference_mode():
 
        # intermediate layers does not use a norm or go through the very last layer of output
        img = transform(image).to(device='cuda',dtype=torch.float32)
        features_out = model.get_intermediate_layers(img, n=[23],reshape=True)    
        features = torch.cat(features_out, dim=1) # B, C, H, W 
    return features.detach().cpu().to(torch.float32).numpy()
def extract_and_process_query(args,image_id_to_sam):
    query_dict = utils.open_file('/home/michal5/dino_sam/coco_retrieval/query_images.pkl')
    model = torch.hub.load(f'facebookresearch/dinov2', f'dinov2_vitl14')
    for category in tqdm(query_dict):
        for image_name in tqdm(query_dict[category]):
            full_image = str(image_name).zfill(12)+'.jpg'
            image = Image.open(os.path.join('/data/common/COCO/train2017',full_image)).convert('RGB')
            features = extract_dino_v2(model,image)
         
            sam_regions = image_id_to_sam[full_image.replace('.jpg','')]
            all_region_features_in_image = []
            # sam regions within an image all have the same total size 
            new_h, new_w = mask_utils.decode(sam_regions[0]['segmentation']).shape
            patch_length = args.dino_patch_length
            padded_h, padded_w = math.ceil(new_h / patch_length) * patch_length, math.ceil(new_w / patch_length) * patch_length # Get the padded height and width
            upsample_feature = torch.nn.functional.upsample(torch.from_numpy(features).cuda(),size=[padded_h,padded_w],mode='bilinear') # First interpolate to the padded size
            upsample_feature = T.CenterCrop((new_h, new_w)) (upsample_feature).squeeze(dim = 0) # Apply center cropping to the original size
            f,h,w = upsample_feature.size()
            for region in sam_regions:
                    sam_region_feature = {}
                    sam_region_feature['region_id'] = region['region_id']
                    sam_region_feature['area'] = region['area']
                    sam_mask = mask_utils.decode(region['segmentation'])
                    r_1, r_2 = np.where(sam_mask == 1)
                    features_in_sam = upsample_feature[:,r_1,r_2].view(f,-1)
                    features_in_sam = torch.max(features_in_sam,dim=1)[0]
                    #features_in_sam = upsample_feature[:,r_1,r_2].view(f,-1).cpu().numpy()
    
                    sam_region_feature['region_feature'] = features_in_sam.cpu().numpy()
                    all_region_features_in_image.append(sam_region_feature)
            utils.save_file(os.path.join(args.region_feature_dir,f'category_{str(category)}',full_image.replace('.jpg','.pkl')),all_region_features_in_image)

def extract_and_process_db(args,image_id_to_sam):
    #query_dict = utils.open_file('/home/michal5/dino_sam/coco_retrieval/query_images.pkl')
    model = torch.hub.load(f'facebookresearch/dinov2', f'dinov2_vitl14')
    all_image_files = os.listdir('/data/common/COCO/val2017')
    for f in tqdm(all_image_files):
            image_name=f.replace('.jpg','')
            full_image = str(image_name).zfill(12)+'.jpg'
            image = Image.open(os.path.join('/data/common/COCO/val2017',full_image)).convert('RGB')
            features = extract_dino_v2(model,image)
            try:
                sam_regions = image_id_to_sam[full_image.replace('.jpg','')]
                all_region_features_in_image = []
                # sam regions within an image all have the same total size 
                new_h, new_w = mask_utils.decode(sam_regions[0]['segmentation']).shape
                patch_length = args.dino_patch_length
                padded_h, padded_w = math.ceil(new_h / patch_length) * patch_length, math.ceil(new_w / patch_length) * patch_length # Get the padded height and width
                upsample_feature = torch.nn.functional.upsample(torch.from_numpy(features).cuda(),size=[padded_h,padded_w],mode='bilinear') # First interpolate to the padded size
                upsample_feature = T.CenterCrop((new_h, new_w)) (upsample_feature).squeeze(dim = 0) # Apply center cropping to the original size
                f,h,w = upsample_feature.size()
                for region in sam_regions:
                        sam_region_feature = {}
                        sam_region_feature['region_id'] = region['region_id']
                        sam_region_feature['area'] = region['area']
                        sam_mask = mask_utils.decode(region['segmentation'])
                        r_1, r_2 = np.where(sam_mask == 1)
                        features_in_sam = upsample_feature[:,r_1,r_2].view(f,-1)
                        features_in_sam = torch.max(features_in_sam,dim=1)[0]
                        #features_in_sam = upsample_feature[:,r_1,r_2].view(f,-1).mean(1).cpu().numpy()
                        #features_in_sam = upsample_feature[:,r_1,r_2].view(f,-1).cpu().numpy()
                        sam_region_feature['region_feature'] = features_in_sam.cpu().numpy()
                        all_region_features_in_image.append(sam_region_feature)
                utils.save_file(os.path.join(args.region_feature_dir,full_image.replace('.jpg','.pkl')),all_region_features_in_image)
            except RuntimeError:
                print('OOM')
                continue 

def load_all_regions(args):
    if len(os.listdir(args.mask_dir)) == 0:
        raise Exception(f"No regions found at {args.mask_dir}")
    print(f"Loading region masks from {args.mask_dir}")
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

    args = parser.parse_args()

    if not args.use_sam:
        print('Using instance masks')
        region_features_vaw(args)
    else:
        print('Using SAM masks')
        image_id_to_mask = load_all_regions(args)
        extract_and_process(args,image_id_to_mask)
        #region_features(args,image_id_to_mask)
    