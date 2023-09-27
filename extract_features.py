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
import utils 
import torch.nn.functional as F
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
def extract_and_upsample(args,model,original_h,original_w,image):
    if args.padding != "center":
        raise Exception("Only padding center is implemented")
    print("Using center padding")
    transform = T.Compose([
        T.ToTensor(),
        lambda x: x.unsqueeze(0),
        CenterPadding(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    with torch.no_grad():
        layers = eval(args.layers)
        if args.intermediate_layers:
            print("Extracting from intermediate layers")
            print(f"Using layers:{eval(args.layers)})")
            features_out = model.get_intermediate_layers(transform(image).cuda(), n=layers,reshape=True)
            features = torch.stack(features_out, dim=-1)
            b,c, h, w, num_layers = features.size()
            if type(layers) == list:
                num_layers = len(layers)
            else:
                num_layers = layers 
            features = features.view(1,c*num_layers,h_14,w_14)
        else:
            print("Using last layers")
            features = model.forward_features(transform(image).cuda())
    return features 

def pool_features(args,features,image_name,image_id_to_sam):
    pooled_features_in_image = []
    sam_regions = image_id_to_sam[image_name.replace('.jpg','')]
    new_h, new_w = mask_utils.decode(sam_regions[0]['segmentation']).shape
    upsample_feature = torch.nn.functional.upsample(features,size=[new_h,new_w],mode='bilinear').squeeze()
    f,h,w = upsample_feature.size()
    for region in sam_regions:
            # try:

            sam_pooled_feature = {}
            sam_pooled_feature['region_id'] = region['region_id']
            sam_pooled_feature['area'] = region['area']
            sam_mask = mask_utils.decode(region['segmentation'])
   
            r_1, r_2 = np.where(sam_mask == 1)
            pooled_region = upsample_feature[:,r_1,r_2].view(f,-1).mean(1).cpu().numpy()
            sam_pooled_feature['pooled_region'] = pooled_region 
            pooled_features_in_image.append(sam_pooled_feature)
    utils.save_pkl_file(os.path.join(args.pooled_dir,args.dataset_name),image_name.replace('.jpg',''),pooled_features_in_image)

def load_all_sam_regions(args):
    print(f"Loading sam regions from {args.sam_location}")
    image_id_to_sam = {}
    for f in tqdm(os.listdir(args.sam_location)):
        sam_regions = utils.open_json_file(args.sam_location,f)
        image_id_to_sam[f.replace('.json','')] = sam_regions
    return image_id_to_sam
def extract_features(model,args):
    all_image_files = [f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
    if len(args.sam_location) == 0:
        raise Exception(f"No sam regions found at {args.sam_location}")
    image_id_to_sam = load_all_sam_regions(args)
    for i,f in enumerate(tqdm(all_image_files,desc='Extract and pool',total=len(all_image_files))):
            image_name = f 
            image = Image.open(os.path.join(self.image_dir,f)).convert('RGB')
            features = extract_and_upsample(args,model,image.size[0],image.size[1],image)
            pool_features(args,features,image_name,image_id_to_sam)
            utils.save_pkl_file(args.feature_dir,image_name,features.cpu().numpy())



