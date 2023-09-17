import pickle
import os 
import numpy as np
import json 
from tqdm import tqdm
from pycocotools import mask as mask_utils
import torch  

class PoolFeatures(object):
    def __init__(self,feature_dir,pooled_dir,sam_location,device='cuda'):
        self.pooled_dir = pooled_dir
        self.feature_dir = feature_dir
        self.sam_location = sam_location 
        self.device = device
        self.image_id_to_sam = None
    def load_all_sam_regions(self):
        image_id_to_sam = {}
        for f in tqdm(os.listdir(self.sam_location)):
            with open(os.path.join(self.sam_location,f),'r+') as sam:
                sam_regions = json.load(sam)
            image_id_to_sam[f.replace('.json','')] = sam_regions
        self.image_id_to_sam = image_id_to_sam
        return image_id_to_sam
    def save_pooled_feature(self,file_name,pooled_features,file_ending='pkl'):
        if file_ending == '.npy':
            with open(os.path.join(self.pooled_dir,file_name),'wb+') as f:
                np.save(f,pooled_features)
        else:
             with open(os.path.join(self.pooled_dir,file_name),'wb+') as f:
                pickle.dump(pooled_features,f)
    def pool_all_features_less_memory(self):
        for i,k in enumerate(tqdm(os.listdir(self.feature_dir),desc='Pooling features',total=len(os.listdir(self.feature_dir)))):
            with torch.no_grad():
                try:
                    feature_vec = np.load(os.path.join(self.feature_dir,k),allow_pickle=True)
                    feature_vec = torch.from_numpy(feature_vec)
                    pooled_features_in_image = []
                    sam_regions = self.image_id_to_sam[k.replace('.npy','')]
                    new_h, new_w = mask_utils.decode(sam_regions[0]['segmentation']).shape
                    upsample_feature = torch.nn.functional.upsample(feature_vec,size=[new_h,new_w],mode='bilinear').squeeze().cuda()
                    for region in sam_regions:
                        sam_pooled_feature = {}
                        sam_pooled_feature['region_id'] = region['region_id']
                        sam_pooled_feature['area'] = region['area']
                        sam_mask = mask_utils.decode(region['segmentation'])
                        r_1, r_2 = np.where(sam_mask == 1)
                        pooled_region = upsample_feature[:,r_1,r_2].view(1024,-1).mean(1).cpu().numpy()
                        sam_pooled_feature['pooled_region'] = pooled_region 
                        pooled_features_in_image.append(sam_pooled_feature)
                    self.save_pooled_feature(k.replace('.npy','.pkl'),pooled_features_in_image)
                except:
                    print(k, 'no region saved')
                    continue 
    def pool_all_features(self):
        for i,k in enumerate(tqdm(os.listdir(self.feature_dir),desc='Pooling features',total=len(os.listdir(self.feature_dir)))):
            with torch.no_grad():
                try:
                    feature_vec = np.load(os.path.join(self.feature_dir,k),allow_pickle=True)
                    feature_vec = torch.from_numpy(feature_vec)
                    pooled_features_in_image = []
                    sam_regions = self.image_id_to_sam[k.replace('.npy','')]
                    new_h, new_w = mask_utils.decode(sam_regions[0]['segmentation']).shape
                    upsample_feature = torch.nn.functional.upsample(feature_vec,size=[new_h,new_w],mode='bilinear').squeeze().cuda()
                    for region in sam_regions:
                        sam_pooled_feature = {}
                        sam_pooled_feature['region_id'] = region['region_id']
                        sam_pooled_feature['area'] = region['area']
                        sam_mask = mask_utils.decode(region['segmentation'])
                        sam_mask = torch.from_numpy(sam_mask).cuda()
                        expanded_mask = sam_mask.expand(upsample_feature.size())
                        pooled_region = upsample_feature[expanded_mask==1].view(1024,-1).mean(1).cpu().numpy()
                        sam_pooled_feature['pooled_region'] = pooled_region 
                        pooled_features_in_image.append(sam_pooled_feature)
                    self.save_pooled_feature(k.replace('.npy','.pkl'),pooled_features_in_image)
                except:
                    print(k, 'no region saved')
                    continue 
# pool = PoolFeatures(feature_dir='/data/michal5/dino_sam/features',pooled_dir='/data/michal5/dino_sam/pooled_features_pkl',sam_location='/shared/rsaas/dino_sam/sam_output/ADE20K/training')
# pool.load_all_sam_regions()
# pool.pool_all_features()