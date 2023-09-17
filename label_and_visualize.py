import numpy as np 
import json 
from PIL import Image, ImageColor
from pycocotools import mask as mask_utils
from tqdm import tqdm
import os 
import cv2
import pickle 
class LabelRegions(object):
     def __init__(self,image_location,annotation_location,sam_location,label_directory,num_labels=150,device='cuda'):
        self.image_location = image_location
        self.annotation_location = annotation_location
        self.sam_location = sam_location 
        self.label_directory = label_directory
        self.device = device 
        self.image_id_to_sam = None 
        self.num_labels = num_labels
     def load_all_sam_regions(self):
        image_id_to_sam = {}
        for f in tqdm(os.listdir(self.sam_location)):
            with open(os.path.join(self.sam_location,f),'r+') as sam:
                sam_regions = json.load(sam)
            image_id_to_sam[f.replace('.json','')] = sam_regions
        self.image_id_to_sam = image_id_to_sam
        return image_id_to_sam
     def label_region(self,sam_region,annotation_map):

        assert sam_region.shape[0] == annotation_map.shape[0]
        assert sam_region.shape[1] == annotation_map.shape[1]
        # find nonzero in sam region
        sam_region_nonzero = np.where(sam_region != 0)
        # get pixel values from map 
        pixel_values_in_region = annotation_map[sam_region_nonzero[0],sam_region_nonzero[1],:].flatten()
        unique_pixels, pixel_counts = np.unique(pixel_values_in_region,return_counts=True)
        all_pixels_in_region = dict(zip(unique_pixels,pixel_counts))

        # get total num of pixels 
        num_pixels = sum(all_pixels_in_region.values())
        #check if any pixel is greater than 95%
        more_than_95 = [pixel_val for pixel_val,pixel_count in all_pixels_in_region.items() if pixel_count>.95*num_pixels]
        # initialize all as None 
        initial_label  = {key: None for key in list(range(1,self.num_labels+1))}
        final_label = {}
        if len(more_than_95)>0:
            # positive for that label 
            assert len(more_than_95)<2 
            final_label[more_than_95[0]] = 1
            # negative for the rest 
            for key in list(range(1,self.num_labels+1)):
                if key != more_than_95[0]:
                    final_label[key] = -1 
        else:
            # all zero 
            final_label = {key:0 for key in list(range(1,self.num_labels+1))}      
        return final_label
     def save_region_labels(self,image_label_dict,file_name):
        with open(os.path.join(self.label_directory,file_name),'wb+') as f:
                pickle.dump(image_label_dict,f)



     def label_all_regions_all_images(self):
        all_images = os.listdir(self.image_location)
        all_annotations = os.listdir(self.annotation_location)
        for i,im in enumerate(tqdm(all_images,desc='Label Features',total=len(all_images))):
            region_to_label = []
            image_name = im 
            annotation_loc = all_annotations[i]
            annotation_map = cv2.imread(os.path.join(self.annotation_location,im.replace('.jpg','.png')))
            sam_regions = self.image_id_to_sam[im.replace('.jpg','')]
            for region in sam_regions:
                sam_labels = {}
                sam_labels['region_id'] = region['region_id']
                sam_mask = mask_utils.decode(region['segmentation'])
                labels  = self.label_region(sam_mask,annotation_map)
                sam_labels['labels'] = labels 
                region_to_label.append(sam_labels)
            self.save_region_labels(region_to_label,image_name.replace('.jpg','.pkl'))
        def map_on_img(self,image_name,mask,color='blue',alpha=0.8):
            img = cv2.imread(os.path.join(self.image_location,name))
            img_to_draw = img.copy()
            nonzero = np.where(mask != 0 )
            img_to_draw[nonzero[0],nonzero[1],:] = np.asarray(ImageColor.getrgb(color))
            img_to_display = alpha*img + (1-alpha)*img_to_draw
            return img_to_display
        def visualize_map(self,annotation_map_loc,color_map_loc):
            color_map = np.load(color_map_loc,allow_pickle=True)
            annotation_map = cv2.imread(os.path.join(self.annotation_location,annotation_map_loc))
            color_seg = np.zeros((first_map.shape[0], first_map.shape[1], 3), dtype=np.uint8)
            for label, color in enumerate(color_map):
                arr_1, arr_2, _ = np.where(first_map == label)
                color_seg[arr_1,arr_2,:] = np.array(color)
            return color_seg 
labels = LabelRegions(image_location='/shared/rsaas/dino_sam/data/ADE20K/images/validation',annotation_location='/shared/rsaas/dino_sam/data/ADE20K/annotations/validation',sam_location='/shared/rsaas/dino_sam/sam_output/ADE20K/validation',label_directory='/shared/rsaas/dino_sam/labels/ADE20K/validation')
labels.load_all_sam_regions()
labels.label_all_regions_all_images()

            




    
