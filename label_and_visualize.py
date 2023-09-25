import numpy as np 
import json 
from PIL import Image, ImageColor
from pycocotools import mask as mask_utils
from tqdm import tqdm
import os 
import scipy
import cv2
import pickle 
from color_maps.pascal_voc import VOC2012_COLORMAP
class LabelRegions(object):
     def __init__(self,annotation_location,sam_location,label_directory,color_map,num_labels=150,device='cuda'):
        self.annotation_location = annotation_location
        self.sam_location = sam_location 
        self.label_directory = label_directory
        self.device = device 
        self.image_id_to_sam = None 
        self.num_labels = num_labels
        self.color_map = color_map
     def load_all_sam_regions(self):
        image_id_to_sam = {}
        for f in tqdm(os.listdir(self.sam_location)):
            with open(os.path.join(self.sam_location,f),'r+') as sam:
                sam_regions = json.load(sam)
            image_id_to_sam[f.replace('.json','')] = sam_regions
        self.image_id_to_sam = image_id_to_sam
        return image_id_to_sam
     def label_region_pascal(self,sam_region,annotation_map):
        sam_region_nonzero = np.where(sam_region != 0)
        rgb_vals = [annotation_map[x,y,:] for x,y in zip(sam_region_nonzero[0],sam_region_nonzero[1])]
        # get pixel values from map
        classes = [np.where(np.all(self.color_map == rgb_vals[-1],axis=-1))[0] for a in rgb_vals] 
        all_classes = [c[0] for c in classes if len(c)!=0]
        unique_pixels, pixel_counts = np.unique(all_classes,return_counts=True)
        all_pixels_in_region = dict(zip(unique_pixels,pixel_counts))
        more_than_95 = [pixel_val for pixel_val,pixel_count in all_pixels_in_region.items()]
        initial_label  = {key: None for key in list(range(1,self.num_labels+1))}
        final_label = {}
        if len(more_than_95)>0 and more_than_95[0]>0:
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
        if not os.path.exists(self.label_directory):
            os.makedirs(self.label_directory)
        with open(os.path.join(self.label_directory,file_name),'wb+') as f:
                pickle.dump(image_label_dict,f)

     def convert_annotation_map(self,annotation_map):
        h, w= annotation_map.shape
        num_classes = 21

        r = annotation_map.copy()
        g = annotation_map.copy()
        b = annotation_map.copy()
        for ll in range(0, 21):
            r[annotation_map == ll] = self.color_map[ll, 0]
            g[annotation_map== ll] = self.color_map[ll, 1]
            b[annotation_map == ll] = self.color_map[ll, 2]
        rgb = np.zeros((annotation_map.shape[0], annotation_map.shape[1], 3))
        rgb[:, :, 0] = r 
        rgb[:, :, 1] = g 
        rgb[:, :, 2] = b 
                
        return rgb


     def label_all_regions_all_images(self):
        
        all_annotations = os.listdir(self.annotation_location)
        for i,ann in enumerate(tqdm(all_annotations,desc='Label Features',total=len(all_annotations))):
            region_to_label = []
            self.color_map = np.array(self.color_map)
            annotation_map =np.array(Image.open(os.path.join(self.annotation_location,ann)))
  
            annotation_map = self.convert_annotation_map(annotation_map)
        
     
            sam_regions = self.image_id_to_sam[ann.replace('.png','')]
            for region in sam_regions:
                sam_labels = {}
                sam_labels['region_id'] = region['region_id']
                sam_mask = mask_utils.decode(region['segmentation'])
                labels  = self.label_region_pascal(sam_mask,annotation_map)
                sam_labels['labels'] = labels 
                region_to_label.append(sam_labels)

            self.save_region_labels(region_to_label,ann.replace('.png','.pkl'))
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
labels = LabelRegions(annotation_location='/shared/rsaas/dino_sam/data/VOCdevkit/VOC2012/segmentation_annotation/train',sam_location='/home/michal5/dino_sam/sam_pascal_voc/train',label_directory='/shared/rsaas/dino_sam/labels/pascal_voc/train',color_map=VOC2012_COLORMAP,num_labels=20)
labels.load_all_sam_regions()
labels.label_all_regions_all_images()

            




    
