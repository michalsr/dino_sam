import os 
import utils
import argparse 
from pycocotools import mask as mask_utils
import numpy as np
from scipy.special import softmax, logit
import scipy
from tqdm import tqdm
from utils import mean_iou
from PIL import Image 
import warnings
warnings.filterwarnings("ignore")
def per_pixel_prediction(args):
    all_pixel_predictions = []
    # keep track of oroder of predictions 
    file_names = []
    model_path = args.classifier_dir
    model_names = [filename for filename in os.listdir(model_path) if filename.startswith("model")]
    val_features = args.val_region_feature_dir
    val_files = [filename for filename in os.listdir(val_features)]
    for file in tqdm(val_files):
        file_names.append(file)
        all_sam = utils.open_file(os.path.join(args.sam_dir,file.replace('.pkl','.json')))
        all_regions = []
        region_order = []
        for i, region in enumerate(all_sam):
            mask = mask_utils.decode(region['segmentation'])
            all_regions.append(mask)
            region_order.append(region['region_id'])
        region_features = utils.open_file(os.path.join(val_features,file))
        feature_all = [area['region_feature'] for j,area in enumerate(region_features)]
        region_all = {area['region_id']:j for j,area in enumerate(region_features)}
        # track region id
        region_idx = [region_all[r] for r in region_order]
        features = np.stack(feature_all)
        features = features[region_idx,:]
        
        if args.classifier_type == 'binary':
            predictions = np.zeros((len(feature_all),args.num_classes+1))
            for i in range(0,args.num_classes+1):
                loaded_model = utils.open_file(os.path.join(model_path,f'model_{i}.sav'))

                preds = loaded_model.decision_function(features)

                predictions[:,i] = preds
        else:
                # should just be one model 
                assert len(model_names) ==1
                loaded_model = utils.open_file(os.path.join(model_path,model_names[0]))
                predictions = loaded_model.decision_function(features)
                assert predictions.shape == len(feature_all),args.num_classes
        if 'after_softmax' in args.multi_region_pixels:
            # averaging softmax values for pixels in multiple regions
            class_predictions = softmax(predictions,axis=1)
        else:
            # use logits for predictions 
            class_predictions = predictions 
        num_regions, num_classes = class_predictions.shape 
        all_regions = np.stack(all_regions,axis=-1)
        h,w,num_regions = all_regions.shape 

        #find pixels where at least one mask equals one
        mask_sum = np.sum(all_regions,axis=-1)
        nonzero_mask = np.nonzero(mask_sum)
        # (h x w x num_regions) x (num_regions,num_classes) --> h x w x num_classes
        # sum h x w across num_regions (for average) 
        product = np.matmul(all_regions[nonzero_mask[0],nonzero_mask[1],:].squeeze(), class_predictions)

        # want avg across softmax values, need to get number of regions summed for each pixel
        # repeat number of regions across softmax values 
        divide = np.repeat(mask_sum[nonzero_mask[0],nonzero_mask[1],np.newaxis],num_classes,axis=1)
        nonzero_region_pixel_preds = np.divide(product,divide)
        if 'before_softmax' in args.multi_region_pixels:
            nonzero_region_pixel_preds = softmax(nonzero_region_pixel_preds,axis=1)
        final_pixel_pred = np.zeros((h,w))
        # index back into original shape
        final_pixel_pred[nonzero_mask[0],nonzero_mask[1]] = np.argmax(nonzero_region_pixel_preds,axis=1)
        all_pixel_predictions.append(final_pixel_pred)
    return all_pixel_predictions, file_names 


def compute_iou(args,predictions,file_names):
    actual_labels = []
    for file in tqdm(file_names):
        actual = np.array(Image.open(os.path.join(args.annotation_dir,file.replace('.pkl','.png'))))
        actual_labels.append(actual)
    if args.ignore_zero:
        num_classes = args.num_classes -1 
        reduce_labels = True
    else:
        num_classes = args.num_classes 
        reduce_labels = False
    miou = mean_iou(results=predictions,gt_seg_maps=actual_labels,num_labels=num_classes,ignore_index=255,reduce_labels=reduce_labels)
    print(miou)
    utils.save_file(os.path.join(args.result_dir,'mean_iou.json'),miou,json_numpy=True)


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
        "--classifier_dir",
        type=str,
        default=None,
        help="Directory containing trained classifier"
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
    parser.add_argument(
        "--sam_dir",
        type=str,
        default=None,
        help="SAM dir"
    )
    parser.add_argument(
        "--val_region_feature_dir",
        type=str,
        default=None,
        help="Location of region features for val"
    )
    parser.add_argument(
        "--multi_region_pixels",
        type=str,
        default="avg_after_softmax",
        help="What to do for pixels in multiple regions. Default is average over probabilities after softmax"
    )
    parser.add_argument("--classifier_type",
    type=str,
    default="binary",
    help="Binary or multi-class ")
    args = parser.parse_args()
    all_pixel_predictions, file_names = per_pixel_prediction(args)
    compute_iou(args,all_pixel_predictions,file_names)






                





    



