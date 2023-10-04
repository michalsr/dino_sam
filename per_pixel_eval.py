import argparse 
from utils import mean_iou
from PIL import Image
from tqdm import tqdm
import utils
import numpy as np
import os

"""
Format predictions and compute mean IOU for a dataset 
"""
def format_predictions(args):
    predicted_labels = []
    actual_labels = []
    all_pixel_files = os.listdir(args.pixel_pred_dir)
    for f in tqdm(all_pixel_files,total=len(all_pixel_files)):
        if '.pkl' in f:
            # load predicted 
            preds = utils.open_file(os.path.join(args.pixel_pred_dir,f))
            actual = np.array(Image.open(os.path.join(args.annotation_dir,f.replace('.pkl','.png'))))
            # load actual
            preds_reshape = np.zeros((actual.shape[0],actual.shape[1]))
            actual_shape = np.zeros((actual.shape[0],actual.shape[1]))
    
        
            for (x,y),v in preds.items():
            
                preds_reshape[x,y] = v
                actual_shape[x,y] = actual[x,y]
            
            # add both 
            predicted_labels.append(preds_reshape)
            actual_labels.append(actual_shape)
    return predicted_labels,actual_labels

def get_mean_iou(args,actual_labels,pred_labels):
    if args.ignore_zero:
        num_classes = args.num_classes -1 
        reduce_labels = True
    else:
        num_classes = args.num_classes 
        reduce_labels = False

    iou_result = mean_iou(
    results=predicted_labels,
    gt_seg_maps=actual_labels,
    num_labels=num_classes,
    ignore_index=255,
    reduce_labels=reduce_labels)
    utils.save_file(os.path.join(args.result_dir,'mean_iou.json'),iou_result,json_numpy=True)
    print(iou_result)
    return iou_result 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_dir",
        type=str,
        default="/shared/rsaas/dino_sam"
    )
    parser.add_argument(
        "--ignore_zero",
        action="store_true",
        help="If want to reduce labels, use this"
    )
    parser.add_argument(
        "--pixel_pred_dir",
        type=str,
        default=None,
        help="Location of class predictions for each pixel "
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="Number of classes in dataset"
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=None,
        help="Location of ground truth annotations"
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Location to store mIOU results"
    )
    
    args = parser.parse_args()
    predicted_labels,actual_labels = format_predictions(args)
    get_mean_iou(args,actual_labels,predicted_labels)