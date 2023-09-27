import os 
from extract_features import extract_features
from region_to_pixel import sam_region_to_pixel, get_region_per_pixel_preds
from train_classifier import train_and_evaluate
import utils 
from per_pixel_inference import format_predictions, get_mean_iou
from sam import get_sam_regions
from args import parse_arguments
import torch 
from label_region import label_all_regions

def full_pipeline(args):
    # check for sam regions 
    if len(os.listdir(os.path.join(args.sam_location,args.dataset_name))) != 0:
        # make sam region
        get_sam_regions(args)
    # check for features 
    if len(os.listdir(os.path.join(args.pool_dir,args.dataset_name))) != 0:
        # extract features 
        model = torch.hub.load(f'{args.model_repo_name}',f'{args.model}')
        extract_features(model,args)
    # check for labels 
    if len(os.listdir(os.path.join(args.label_dir,args.dataset_name))) != 0:
        # label 
        label_all_regions(args)
    # check for training 
    if len(os.listdir(os.path.join(args.classifier_dir,args.dataset_name))) != 0:
        # train and eval 
        train_and_evaluate(args)
    # get predictions for each pixel 
    if len(os.listdir(os.path.join(args.pixel_region_id,args.dataset_name))) != 0:
        sam_region_to_pixel(args)
        get_region_per_pixel_preds(args)
    # inference 
    predicted_labels,actual_labels = format_predictions(args)
    get_mean_iou(predicted_labels,actual_labels)




if __name__ == '__main__':
    args = parse_arguments()
    full_pipeline(args)