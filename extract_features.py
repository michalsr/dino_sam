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
For extraction features for a given dataset and model. 
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
    
def extract_dino_v1(args,model,image):
    if args.padding != "center":
        raise Exception("Only padding center is implemented")
    print("Using center padding")
    transform = T.Compose([
        T.ToTensor(),
        lambda x: x.unsqueeze(0),
        CenterPadding(multiple = args.multiple),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    with torch.no_grad():
        layers = eval(args.layers)
     
        print(f"Using layers:{layers}")
        # intermediate layers does not use a norm or go through the very last layer of output
        model = model.cuda()
        features_out = model.get_intermediate_layers(transform(image).cuda(), n=layers)
        features = [f[:, 1:] for f in features_out] # Remove the cls tokens
        features = torch.cat(features_out, dim=1) # B, C, H * W
        B, C, _, _= features.shape
        H, W = image.size
        patch_H, patch_W = math.ceil(H / args.multiple), math.ceil(W / args.multiple)
        features = features.view(B, C, patch_H, patch_W) 
    return features 

def extract_dino_v2(args,model,image):
    if args.padding != "center":
        raise Exception("Only padding center is implemented")
    print("Using center padding")
    transform = T.Compose([
        T.ToTensor(),
        lambda x: x.unsqueeze(0),
        CenterPadding(multiple = args.multiple),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    with torch.no_grad():
        layers = eval(args.layers)
     
        print(f"Using layers:{layers}")
        # intermediate layers does not use a norm or go through the very last layer of output
        model = model.cuda()
        features_out = model.get_intermediate_layers(transform(image).cuda(), n=layers,reshape=True)
        features = torch.cat(features_out, dim=1) # B, C, H, W 
    return features 


def extract_features(model,args):
    all_image_files = [f for f in os.listdir(args.image_dir) if os.path.isfile(os.path.join(args.image_dir, f))]
    for i,f in enumerate(tqdm(all_image_files,desc='Extract',total=len(all_image_files))):
        image_name = f 
        filename_extension = os.path.splitext(image_name)[1]  
        image = Image.open(os.path.join(args.image_dir,f)).convert('RGB')
        if 'dino' in args.model:
            if 'dinov2' in args.model:
                features = extract_dino_v2(args,model,image)
            else: # dinov1
                features = extract_dino_v1(args,model,image)
        utils.save_file(os.path.join(args.feature_dir,image_name.replace(filename_extension,".pkl")),features.cpu().numpy())



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
        help="Location of feature files",
    )
    parser.add_argument(
        "--model_repo_name",
        type=str,
        default="facebookresearch/dinov2",
        choices=['facebookresearch/dinov2','facebookresearch/dino:main'],
        help="PyTorch model name for downloading from PyTorch hub"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='dinov2_vitl14',
        choices=['dinov2_vitl14','dino_vitb8'],
        help="Name of model from repo"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="[20,21,22,23]",
        help="List of layers or number of last layers to take"
    )
    parser.add_argument(
        "--padding",
        default="center",
        help="Padding used for transforms"
    )
    parser.add_argument(
        "--multiple",
        type=int,
        default=14,
        help="The patch length of the model"
    )
    
    args = parser.parse_args()
    model = torch.hub.load(f'{args.model_repo_name}',f'{args.model}')
    extract_features(model,args)