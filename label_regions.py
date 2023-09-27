import numpy as np 
import json 
from PIL import Image, ImageColor
from pycocotools import mask as mask_utils
from tqdm import tqdm
import os 
import scipy
import cv2
import pickle 
import utils 


def load_all_sam_regions(args):
    print(f"Loading sam regions from {args.sam_location}")
    image_id_to_sam = {}
    for f in tqdm(os.listdir(args.sam_location)):
        sam_regions = utils.open_json_file(args.sam_location,f)
        image_id_to_sam[f.replace('.json','')] = sam_regions
    return image_id_to_sam



def label_region(args,sam_region,annotation_map):
    sam_region_nonzero = np.where(sam_region != 0)
    # get pixel values from map 
    pixel_values_in_region = annotation_map[sam_region_nonzero[0],sam_region_nonzero[1]].flatten()
    unique_pixels, pixel_counts = np.unique(pixel_values_in_region,return_counts=True)
    all_pixels_in_region = dict(zip(unique_pixels,pixel_counts))

    # get total num of pixels 
    num_pixels = sum(all_pixels_in_region.values())
    #check if any pixel is greater than certain percent value 
    more_than_percent= [pixel_val for pixel_val,pixel_count in all_pixels_in_region.items() if pixel_count>(args.label_percent/100)*num_pixels]
    # initialize all as None 
    start_class = 0
    if args.ignore_zero:
        start_class = 1
    initial_label  = {key: None for key in list(range(start_class,args.num_classes+1))}
    final_label = {}

    if len(more_than_percent)>0:
        # positive for that label 
        assert len(more_than_percent)<2 
        final_label[more_than_percent[0]] = 1
        # negative for the rest 
        for key in list(range(start_class,args.num_classes+1)):
            if key != more_than_percent[0]:
                final_label[key] = -1 
    else:
        # all zero 
        final_label = {key:0 for key in list(range(start_class,args.num_classes+1))}      
    return final_label

def label_all_regions(args):
    if len(args.sam_location) == 0:
        raise Exception(f"No sam regions found at {args.sam_location}")
    image_id_to_sam = load_all_sam_regions(args)
    all_annotations = os.listdir(args.annotation_location)
    for i,ann in enumerate(tqdm(all_annotations,desc='Label Features',total=len(all_annotations))):
        region_to_label = []
        annotation_map =np.array(Image.open(os.path.join(args.annotation_location,ann)))
        sam_regions = image_id_to_sam[ann.replace('.png','')]
        for region in sam_regions:
            sam_labels = {}
            sam_labels['region_id'] = region['region_id']
            sam_mask = mask_utils.decode(region['segmentation'])
            labels = label_region(args,sam_region,annotation_map)
            sam_labels['labels'] = labels 
            region_to_label.append(sam_labels)
        utils.save_pkl_file(os.path.join(args.region_labels,args.dataset_name),ann.replace('.png',''),region_to_label)



