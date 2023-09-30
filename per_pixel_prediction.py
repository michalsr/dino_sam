import numpy as np 
import json
import os 
import warnings
import pickle
import argparse
from tqdm import tqdm 
import scipy
from scipy.special import softmax, logit
from pycocotools import mask as mask_utils
import utils 
"""
Label each region and each pixel within each region
"""
def sam_region_to_pixel(args):
    # get pixel values for each region
    all_sam = os.listdir(args.sam_dir)
    for f in tqdm(all_sam):
        # open sam file 
        all_pixels = {}
        sam_regions = utils.open_json_file(os.path.join(sam_dir,f))
        for region in sam_regions:
            mask = mask_utils.decode(region['segmentation'])
            #can use np.nonzero as well 
            nonzero = np.where(mask==1)
            for x,y in zip(nonzero[0],nonzero[1]):
                    assert mask[x,y] == 1
                    if (x,y) not in all_pixels:
                        all_pixels[(x,y)] = []
                    all_pixels[(x,y)].append(region['region_id'])
        utils.save_pkl_file(os.path.join(args.pixel_to_region_dir,f.replace('.json','.pkl')),all_pixels)

def get_region_per_pixel_preds(args):
    model_path = args.classifier_dir
    model_names = [filename for filename in os.listdir(model_path) if filename.startswith("model")]
    val_feature = args.val_region_feature_dir
    pixel_region = args.pixel_to_region_dir
    files = [filename for filename in os.listdir(pixel_region)]
    for file in tqdm(files):
        all_pixels = {}
        region_features = utils.open_pkl_file(os.path.join(val_feature,file))
        if args.ignore_zero:
            total_classes -= 1 
            min_class = 1
        else:
            total_classes = args.num_classes
            min_class = 0
        feature_predictions = np.zeros((total_classes, len(region_features)))
        feature_all = [area['region_feature'] for j,area in enumerate(region_features)]
        region_all = {area['region_id']:j for j,area in enumerate(region_features)}
        features = np.stack(feature_all)
        for i in range(min_class,args.num_classes+1):
            loaded_model = utils.open_pkl_file(os.path.join(model_path,f'model_{i}.sav'))
            predictions = loaded_model.decision_function(features)
            if args.ignore_zero:
                feature_predictions[i-1, :] = predictions
            else:
                feature_predictions[i, :] = predictions
        pixel_to_region_map = utils.open_pkl_file(os.path.join(pixel_region,file))
        all_features = softmax(feature_predictions,axis=0)
        all_pixels = {k:np.argmax(all_features[:,[region_all[x] for x in v]].mean(axis=1),axis=0) for k,v in pixel_to_region_map.items()}
        utls.save_pkl_file(os.path.join(args.pixel_pred_dir,file,)all_pixels)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_dir",
        type=str,
        default="/shared/rsaas/dino_sam"
    )
    parser.add_argument(
        "--val_region_feature_dir",
        type=str,
        default=None,
        help="Location of region features for val"
    )
    parser.add_argument(
        "--num_classes",
        default=0,
        help="Number of classes in dataset"
    )
    parser.add_argument(
        "--pixel_to_region_dir",
        type=str,
        default=None,
        help="Location to store maps from pixels in each image to region ids"
    )
    parser.add_argument(
        "--classifier_dir",
        type=str,
        default=None,
        help="Location to store trained classifiers"
    )
    parser.add_argument(
        "--sam_dir",
        type=str,
        default=None,
        help="SAM dir"
    )
    parser.add_argument(
        "--ignore_zero",
        action="store_true"
        help="Include 0 class"
    )
    parser.add_argument(
        "--pixel_pred_dir",
        type=str,
        default=None,
        help="Location to store class predictions for each pixel"
    )
    args = parser.parse_args()
    sam_region_to_pixel(args)
    get_region_per_pixel_preds(args)

    