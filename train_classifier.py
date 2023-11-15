from sklearn.linear_model import LogisticRegression
import pickle
import os
import numpy as np
from tqdm import tqdm
import argparse 
import utils 
import gc
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from pycocotools import mask as mask_utils
from scipy.special import softmax, logit
import scipy
from tqdm import tqdm
from utils import mean_iou
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
"""
Train and eval classifiers 
"""

def per_pixel_prediction(args):
    all_pixel_predictions = []
    # keep track of oroder of predictions 
    file_names = []
    model_path = args.classifier_dir
    #model_path = os.path.join(args.classifier_dir,'classifier')
    #model_path = os.path.join(args.classifier_dir,args.classifier_name)
    model_names = [filename for filename in os.listdir(model_path) if filename.startswith("model")]
    val_features = args.val_region_feature_dir
    val_files = [filename for filename in os.listdir(val_features)]
    for file in tqdm(val_files):
        file_names.append(file)
        all_sam = utils.open_file(os.path.join(args.sam_dir,file.replace('.pkl','.json')))
        if args.use_pos_embd:
            pos_embd = utils.open_file(os.path.join(args.val_pos_embd_dir,file))
        all_regions = []
        region_order = []
       
        for i, region in enumerate(all_sam):
            mask = mask_utils.decode(region['segmentation'])
            all_regions.append(mask)
            region_order.append(region['region_id'])
        region_features = utils.open_file(os.path.join(val_features,file))
        if args.use_pos_embd:
            feature_all = []
            for j,area in enumerate(region_features):
                 feature_all.append(area['region_feature']+pos_embd[i,:])
        else:

            feature_all = [area['region_feature'] for j,area in enumerate(region_features)]
        region_all = {area['region_id']:j for j,area in enumerate(region_features)}
        # track region id
        region_idx = [region_all[r] for r in region_order]
        features = np.stack(feature_all)
        features = features[region_idx,:]
        

        assert len(model_names) ==1
        loaded_model = utils.open_file(os.path.join(model_path,model_names[0]))
        if args.mlp:
            predictions= loaded_model.predict_proba(features)
        else:
            predictions = loaded_model.decision_function(features)
       
        assert predictions.shape == (len(feature_all),args.num_classes+1)
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
    # num classes = highest index
    if args.ignore_zero:
        num_classes = args.num_classes -1
        reduce_labels = True
        reduce_pred_labels = True 
    else:
        num_classes = args.num_classes +1
        reduce_labels = False
        reduce_pred_labels=False 
    if args.ade==True:
        assert reduce_labels==True 
        assert reduce_pred_labels==True
        assert num_classes == 149 
    miou = mean_iou(results=predictions,gt_seg_maps=actual_labels,num_labels=num_classes,ignore_index=255,reduce_labels=reduce_labels,reduce_pred_labels=reduce_pred_labels)
    print(miou)
    utils.save_file(os.path.join(args.results_dir,'mean_iou.json'),miou,json_numpy=True)

def train_multi_classifier(args):
    file = os.path.join(args.classifier_dir, 'train.pkl')
    train_data = utils.open_file(file)
    target_label = train_data['label']
    if args.ade:
        class_weight = {x:1 for x in range(151)}
        class_weight[0] = 0
        classifier = LogisticRegression(verbose=1, multi_class='multinomial', max_iter=args.iterations,class_weight=class_weight).fit(train_data['feature'],
                                                                                                        target_label,
                                                                                    sample_weight=train_data['weight'])

    else:
        classifier = LogisticRegression(verbose=1, multi_class='multinomial', max_iter=args.iterations).fit(train_data['feature'],
                                                                                                        target_label,
                                                                                    sample_weight=train_data['weight'])
    save_dir = os.path.join(args.classifier_dir,f'model_multiclass.sav')
    utils.save_file(save_dir,classifier)
    print(f'Saved multiclass clssifier')
    all_pixel_predictions, file_names = per_pixel_prediction(args)
    

    compute_iou(args,all_pixel_predictions,file_names)




def load_features(args,split='train'):
    # assumes that the first entry in label dict is the positive class. If not use load_features_vaw
    feature_all = []
    label_all = []
    weight_all = []
    if split == 'train':
        feature_dir = args.train_region_feature_dir 
        label_dir = args.train_region_labels_dir 
        if args.use_pos_embd:
            pos_embed_dir = args.train_pos_embd_dir
        else:
            pos_embd_dir=None
    else:
        feature_dir = args.val_region_feature_dir
        label_dir = args.val_region_labels_dir 
        if args.use_pos_embd:
            pos_embed_dir = args.val_pos_embd_dir



    label_files = os.listdir(label_dir)

    for file_name in tqdm(label_files):
        if '.pkl'  not in file_name:
            file_name = file_name + '.pkl'
        try:
            file_features = utils.open_file(os.path.join(feature_dir,file_name))
            if pos_embd_dir != None:

                pos_embd = utils.open_file(os.path.join(pos_embd_dir,file_name))

        except:
            continue 
        file_labels = utils.open_file(os.path.join(label_dir,file_name))
      

        for i, area in enumerate(file_features):
            if pos_embd_dir != None:
                area_feature = area['region_feature']+pos_embd[i,:]
            else:
                area_feature = area['region_feature']
            area_label = file_labels[i]['labels']
            target_label = list(area_label.keys())[0]

            if area_label[target_label] == 1:
                feature_all.append(area_feature)
                label_all.append(target_label)
                weight_all.append(area['area'])

    save_dict = {}
    save_dict['feature'] = np.stack(feature_all)
    save_dict['label'] = np.stack(label_all)
    save_dict['weight'] = np.stack(weight_all)
    utils.save_file(os.path.join(args.classifier_dir,split+'.pkl'),save_dict)
    return save_dict

def train_mlp(args):

    clf = MLPClassifier(verbose=True,learning_rate_init=args.mlp_lr,max_iter=args.iterations,hidden_layer_sizes=[1000],early_stopping=True,n_iter_no_change=5)
    if args.use_pos_embd:
        train_data = utils.open_file(os.path.join(args.classifier_dir, 'train_pos_embd.pkl'))
    else:
        train_data = utils.open_file(os.path.join(args.classifier_dir, 'train.pkl'))
   
    train_features = train_data['feature']
    target_label = train_data['label']
    clf.fit(train_data['feature'],train_data['label'])
    save_dir = os.path.join(args.classifier_dir,f'model_mlp.sav')
    utils.save_file(save_dir,clf)
    print(f'Saved classifier {save_dir}')
    clf = utils.open_file(os.path.join(args.classifier_dir, f'model_mlp.sav'))
    if args.use_pos_embd:
        val_data = utils.open_file(os.path.join(args.classifier_dir,'val_pos_embd.pkl'))
    else:

        val_data = utils.open_file(os.path.join(args.classifier_dir,'val.pkl'))
        
    val_features = val_data['feature']
    val_label = val_data['label']
    all_pixel_predictions, file_names = per_pixel_prediction(args)
    compute_iou(args,all_pixel_predictions,file_names)
    


def train_and_evaluate(args):
    if args.num_classes != 150 and args.num_classes!= 20:
        raise ValueError('ADE should have 150 and Pascal VOC should have 21')
    if args.num_classes == 150:
        if args.ade ==False:
            raise ValueError('If using ADE then ade argument should be set to True')
    if args.ade==True:
        print('Training and evaluating on ADE. Make sure to use the correct region label directory (ADE20K_no_zero)!')
    if args.eval_only:
        all_pixel_predictions, file_names = per_pixel_prediction(args)
        compute_iou(args,all_pixel_predictions,file_names)
    else:
        if args.use_pos_embd:
            training_file = os.path.join(args.classifier_dir,'train_pos_embd.pkl')
            val_file = os.path.join(args.classifier_dir,'val_pos_embd.pkl')
        else:
            training_file = os.path.join(args.classifier_dir,'train.pkl')
            val_file = os.path.join(args.classifier_dir,'val.pkl')
        
    
        # # root_feature, root_label, save_root,
        if not os.path.exists(training_file):
            train_data = load_features(args,'train')
        
        avg_ap = []
        avg_roc_auc = []
        if args.mlp:
            if not args.eval_only:
                train_mlp(args)
        else: 
            train_multi_classifier(args)

      
    all_pixel_predictions, file_names = per_pixel_prediction(args)
    if args.output_predictions:
        print('Saving predictions to PNGs')
        prediction_dir = os.path.join(args.results_dir, 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)

        for file_name, prediction in tqdm(zip(file_names, all_pixel_predictions)):
            prediction = Image.fromarray(prediction.astype(np.uint8))
            prediction.save(os.path.join(prediction_dir, file_name.replace('.pkl', '.png')))
    compute_iou(args,all_pixel_predictions,file_names)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_dir",
        type=str,
        default="/shared/rsaas/dino_sam"
    )
    parser.add_argument(
        "--train_region_labels_dir",
        type=str,
        default=None,
        help="Location where ground truth label regions are stored for training images",
    )
    parser.add_argument(
        "--val_region_labels_dir",
        type=str,
        default=None,
        help="Location where ground truth label regions are stored for val images",
    )
    parser.add_argument('--iterations',
                        type=int,
                         default=1000,
                         help='Number of iterations to run log regression')
    parser.add_argument(
        "--ignore_zero",
        action="store_true",
        help="Include 0 class"
    )
    parser.add_argument(
        "--train_region_feature_dir",
        type=str,
        default=None,
        help="Location of features for each region in training images"
    )
    parser.add_argument(
        "--val_region_feature_dir",
        type=str,
        default=None,
        help="Location of features for each region in val images"
    )

    parser.add_argument(
        "--classifier_dir",
        type=str,
        default=None,
        help="Location to store trained classifiers"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Location to store AP, AUC-ROC results"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="No classifier training"
    )

    parser.add_argument(
        "--use_weight",
        action="store_false",
        help="Whether to use area as weight."
    )

    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="To use the balance class option in scikit learn for unbalanced classes"
    )

    parser.add_argument(
        "--mlp",
        action="store_true",
        help="Use mlp"
    )
    parser.add_argument("--ade",action="store_true")
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=.0001,
        help="Use mlp"
    )

    parser.add_argument(
        "--sam_dir",
        type=str,
        default=None,
        help="SAM masks for eval"
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default=None,
        help="Location of ground truth annotations"
    )
    parser.add_argument(
        "--multi_region_pixels",
        type=str,
        default="avg_after_softmax",
        help="What to do for pixels in multiple regions. Default is average over probabilities after softmax"
    )

    parser.add_argument("--use_pos_embd",
    action="store_true",
    help="Add in sam pos encod")
    parser.add_argument("--train_pos_embd_dir",
    type=str,
    default=None,
    help="Train pos embedding")
    parser.add_argument("--val_pos_embd_dir",
    type=str,
    default=None,
    help="Val pos embedding")

    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="Number of classes in dataset"
    )
    parser.add_argument(
        "--output_predictions",
        action='store_true',
    )
    args = parser.parse_args()
    train_and_evaluate(args)