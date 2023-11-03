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
import torch 
import torchvision.transforms as T
import argparse
import torch.nn.functional as F

def load_all_sam_regions(args):
    print(f"Loading sam regions from {args.sam_dir}")
    image_id_to_sam = {}
    for f in tqdm(os.listdir(args.sam_dir)):
        sam_regions = utils.open_file(os.path.join(args.sam_dir,f))
        image_id_to_sam[f.replace('.json','')] = sam_regions
    return image_id_to_sam



def label_region(args,sam_region,annotation_map):
    sam_region_nonzero = np.where(sam_region != 0)
    # get pixel values from map 

    pixel_values_in_region = annotation_map[sam_region_nonzero[0],sam_region_nonzero[1]].flatten()
    unique_pixels, pixel_counts = np.unique(pixel_values_in_region,return_counts=True)
    all_pixels_in_region = dict(zip(unique_pixels,pixel_counts))
    start_class = 0
    if args.ignore_zero:
        start_class = 1
    # get total num of pixels 
    num_pixels = sum(all_pixels_in_region.values())
    #check if any pixel is greater than certain percent value 
    more_than_percent= [(pixel_val,pixel_count) for pixel_val,pixel_count in all_pixels_in_region.items() if all((pixel_count>((args.label_percent/100)*num_pixels),pixel_val>=start_class,pixel_val<=args.num_classes+1))]
    # initialize all as None 

    initial_label  = {key: None for key in list(range(start_class,args.num_classes+1))}
    final_label = {}


    if len(more_than_percent)>0:
        max_idx = np.argmax(np.asarray([t[1] for t in more_than_percent]))
        max_pixel_class = [t[0] for t in more_than_percent][max_idx]
        # positive for that label 

        final_label[max_pixel_class] = 1
        # negative for the rest 
        for key in list(range(start_class,int(args.num_classes)+1)):
            if key != max_pixel_class:
                final_label[key] = -1 
    else:
        # all zero 
        final_label = {key:0 for key in list(range(start_class,int(args.num_classes)+1))}      
    return final_label
def label_vaw(args):
    # Already have regions from data
    # each entry has image id, polygon coordinates
    annotation_file = utils.open_file(os.path.join(args.annotation_dir,args.annotation_file))
   
    # annotation file uses actual class names not numbers
    label_map = utils.open_file(args.label_map)
    # map images to instances to save space 
    all_image_ids = [entry['image_id'] for entry in annotation_file]

    image_to_instances = {key:[] for key in all_image_ids}
    
    for entry in tqdm(annotation_file):
        instance_dict = {}
        instance_dict['instance_id'] = entry['instance_id']

         # 620 attributes 
        instance_labels  = {key: -1 for key in list(range(0,621))}
        if not args.use_sam:
            # convert polygon to region 
            # file name is instance id 
            img = np.array(Image.open(os.path.join(args.image_dir,entry['image_id']+'.jpg')))

            h, w = img.shape[0],img.shape[1] 
            if entry['instance_polygon'] != None:
                mask = utils.polygon_to_mask(entry['instance_polygon'],h,w)
                instance_dict['mask'] = mask
            else:
                continue 
            
            # just read attributes from file 
            for p in entry['positive_attributes']:
                idx = label_map[p]
                instance_labels[idx] = 1
        instance_dict['labels'] = instance_labels
        image_to_instances[entry['image_id']].append(instance_dict)
    for image_id in tqdm(image_to_instances):
        utils.save_file(os.path.join(args.region_labels,image_id+'.pkl'),image_to_instances[image_id])

    




       


def label_all_regions(args):
    if len(os.listdir(args.sam_dir)) == 0:
        raise Exception(f"No sam regions found at {args.sam_dir}")
    image_id_to_sam = load_all_sam_regions(args)
    all_annotations = os.listdir(args.annotation_dir)
    for i,ann in enumerate(tqdm(all_annotations,desc='Label Features',total=len(all_annotations))):
        region_to_label = []
        annotation_map =np.array(Image.open(os.path.join(args.annotation_dir,ann)),dtype=np.int64)
      
        sam_regions = image_id_to_sam[ann.replace('.png','')]
        for region in sam_regions:
            sam_labels = {}
            sam_labels['region_id'] = region['region_id']
            sam_mask = mask_utils.decode(region['segmentation'])
            labels = label_region(args,sam_mask,annotation_map)
            sam_labels['labels'] = labels 
            region_to_label.append(sam_labels)
        utils.save_file(os.path.join(args.region_labels,ann.replace('.png','.pkl')),region_to_label)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_dir",
        type=str,
        default="/shared/rsaas/dino_sam"
    )
    parser.add_argument(
        "--region_labels",
        type=str,
        default=None,
        help="Location to store ground truth of label regions",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=None,
        help="Location of per-pixel annotations",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default=None,
        help="If annotations do not match up with image ids",
    )
    parser.add_argument(
        "--label_map",
        type=str,
        default=None,
        help="Convert class names to class ids",
    )
    parser.add_argument(
        "--vaw",
        action="store_true",
        help="If Visual Attributes in the Wild (VAW) dataset. Different fxn used",
    )
    parser.add_argument(
        "--use_sam",
        action="store_false",
        help="Only for VAW. Whether to use SAM regions or default instance masks",
    )
    parser.add_argument(
        "--ignore_zero",
        action="store_true",
        help="Include 0 class"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="Number of classes in dataset"
    )
    parser.add_argument(
        "--sam_dir",
        type=str,
        default=None,
        help="Location of SAM regions"
    )
    parser.add_argument(
        "--label_percent",
        type=int,
        default=50,
        help="Percent of pixels within a region that need to belong to the same class before region is assigned that label"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Image dir for vaw if needed for saving masks")
    
    args = parser.parse_args()
    if args.vaw:
        print('Labeling VAW dataset')
        label_vaw(args)
    else:
        label_all_regions(args)