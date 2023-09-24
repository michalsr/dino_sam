# pool and extract features at the same time 
# assumes sam masks exist 
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

class PoolExtractFeatures(object):
    def __init__(self,image_dir,pooled_dir,sam_location,feature_dir,padding,model_name='dinov2',model_type='dinov2_vitl14',device='cuda',layers=4):
        self.pooled_dir = pooled_dir
        self.image_dir = image_dir
        self.sam_location = sam_location 
        self.device = device
        self.image_id_to_sam = None
        self.model_name = model_name
        self.model_type = model_type 
        self.layers = layers
        self.feature_dir = feature_dir 
        self.model = None 
        self.padding = padding 
    def load_all_sam_regions(self):
        image_id_to_sam = {}
        for f in tqdm(os.listdir(self.sam_location)):
            with open(os.path.join(self.sam_location,f),'r+') as sam:
                sam_regions = json.load(sam)
            image_id_to_sam[f.replace('.json','')] = sam_regions
        self.image_id_to_sam = image_id_to_sam
        return image_id_to_sam
    def extract_and_upsample_features(self,original_h,original_w,image):
   
        transform = T.Compose([
        T.ToTensor(),
        lambda x: x.unsqueeze(0),
        self.padding,
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
    
        with torch.no_grad():
            features_out = self.model.get_intermediate_layers(transform(image).cuda(), n=self.layers,reshape=True)
        features = torch.stack(features_out, dim=-1)
        
    
        b,c, h_14, w_14, num_layers = features.size()
        features = features.view(1,c*len(self.layers),h_14,w_14)

        
    
        return features 
    def pool_features(self,feature,image_name):
        pooled_features_in_image = []
        sam_regions = self.image_id_to_sam[image_name.replace('.jpg','')]
        new_h, new_w = mask_utils.decode(sam_regions[0]['segmentation']).shape
        upsample_feature = torch.nn.functional.upsample(feature,size=[new_h,new_w],mode='bilinear').squeeze()
        for region in sam_regions:
            # try:
            sam_pooled_feature = {}
            sam_pooled_feature['region_id'] = region['region_id']
            sam_pooled_feature['area'] = region['area']
            sam_mask = mask_utils.decode(region['segmentation'])
   
            r_1, r_2 = np.where(sam_mask == 1)
            pooled_region = upsample_feature[:,r_1,r_2].view(1024,-1).mean(1).cpu().numpy()
            sam_pooled_feature['pooled_region'] = pooled_region 
            pooled_features_in_image.append(sam_pooled_feature)
               
            # except:
            #     print(image_name, 'no region saved')
            #     continue 
        self.save_features(pooled_features_in_image,image_name.replace('.jpg',''),self.pooled_dir)

    def save_features(self,features,filename,parent_dir):
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    
        # Construct the full path for the .pkl file
       # Remove the file extension
        pkl_filename = f"{filename}.pkl"
        pkl_path = os.path.join(parent_dir, pkl_filename)
        
        with open(pkl_path, 'wb') as f:
            pickle.dump(features, f)
    
    def extract_pool_all(self):
        self.model = torch.hub.load(f'facebookresearch/{self.model_name}', self.model_type).cuda().eval()
        all_image_files =  [f for f in os.listdir(self.image_dir) if os.path.isfile(os.path.join(self.image_dir, f))]
        self.load_all_sam_regions()
        for i,f in enumerate(tqdm(all_image_files,desc='Extract and pool',total=len(all_image_files))):
            image_name = f 
            image = Image.open(os.path.join(self.image_dir,f)).convert('RGB')

            feature = self.extract_and_upsample_features(image.size[0],image.size[1],image)
            self.pool_features(feature,image_name)
            self.save_features(feature.cpu().numpy(),image_name,self.feature_dir)
           


extract_and_upsample = PoolExtractFeatures('/data/michal5/segmentation_imgs/val','/home/michal5/dino_sam/pooled_features_pkl/pascal_voc/val','/home/michal5/dino_sam/sam_pascal_voc/val','/home/michal5/dino_sam/features_pkl/pascal_voc/val',CenterPadding(),layers=[20, 21, 22, 23])
extract_and_upsample.extract_pool_all()



            
    

