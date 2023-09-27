import numpy as np 
import json
import os 
import warnings
import pickle
from tqdm import tqdm 
import scipy
from scipy.special import softmax, logit
from pycocotools import mask as mask_utils
import utils 
def sam_region_to_pixel(args):
    # get pixel values for each region
    sam_dir = os.path.join(args.sam_location,args.dataset_name,'val')
    all_sam = os.listdir(sam_dir)
    for f in tqdm(all_sam):
        # open sam file 
        all_pixels = {}
        sam_regions = utils.open_json_file(sam_dir,f)
        for region in sam_regions:
            mask = mask_utils.decode(region['segmentation'])
            #can use np.nonzero as well 
            nonzero = np.where(mask==1)
            for x,y in zip(nonzero[0],nonzero[1]):
                    assert mask[x,y] == 1
                    if (x,y) not in all_pixels:
                        all_pixels[(x,y)] = []
                    all_pixels[(x,y)].append(region['region_id'])
        utils.save_pkl_file(os.path.join(args.main_dir,'pixel_region_id',args.dataset_name),f.replace('.json','.pkl'),all_pixels)

def get_region_per_pixel_preds(args):
    model_path = os.path.join(args.classifier_dir,args.datset_name)
    model_names = [filename for filename in os.listdir(model_path) if filename.startswith("model")]
    val_feature = os.path.join(args.pooled_dir,args.dataset_name,'val')
    pixel_region = os.path.join(args.main_dir,'pixel_region_id',args.dataset_name)
    files = [filename for filename in os.listdir(pixel_region)]
    for file in tqdm(files):
        all_pixels = {}
        data = utils.open_pkl_file(val_feature,file)
        if args.ignore_zero:
            total_classes -= 1 
            min_class = 1
        else:
            total_classes = args.num_classes
            min_class = 0
        feature_predictions = np.zeros((total_classes, len(data)))
        feature_all = [area['pooled_region'] for j,area in enumerate(data)]
        region_all = {area['region_id']:j for j,area in enumerate(data)}
        features = np.stack(feature_all)
        for i in range(min_class,args.num_classes+1):
            loaded_model = utils.open_pkl_file(model_path,f'model_{i}.sav')
            predictions = loaded_model.decision_function(features)
            if args.ignore_zero:
                feature_predictions[i-1, :] = predictions
            else:
                feature_predictions[i, :] = predictions
        data2 = utils.open_pkl_file(pixel_region,file)
        all_features = softmax(feature_predictions,axis=0)
        all_pixels = {k:np.argmax(all_features[:,[region_all[x] for x in v]].mean(axis=1),axis=0) for k,v in data2.items()}
        utls.save_pkl_file(os.path.join(args.main_dir,'feature_preds',args.dataset_name),file,all_pixels)
