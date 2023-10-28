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
    if args.ignore_zero:
        num_classes = args.num_classes -1 
        reduce_labels = True
    else:
        num_classes = args.num_classes 
        reduce_labels = False
    miou = mean_iou(results=predictions,gt_seg_maps=actual_labels,num_labels=num_classes,ignore_index=255,reduce_labels=reduce_labels)
    utils.save_file(os.path.join(args.result_dir,'mean_iou.json'),miou,json_numpy=True)



def load_features_vaw(args,split='train'):
    # multiple classes have value of 1 
    feature_all = []
    label_all = []
    weight_all = []
    if split == 'train':
        feature_dir = args.train_region_feature_dir 
        label_dir = args.train_region_labels_dir 
    else:
        feature_dir = args.val_region_feature_dir
        label_dir = args.val_region_labels_dir 


    label_files = os.listdir(label_dir)
    for file_name in tqdm(label_files):
        if '.pkl'  not in file_name:
            file_name = file_name + '.pkl'
        file_features = utils.open_file(os.path.join(feature_dir,file_name))
        file_labels = utils.open_file(os.path.join(label_dir,file_name))

        for i, area in enumerate(file_features):
            area_feature = area['region_feature']
            area_label = file_labels[i]['labels']
            # labels should be array of total number of labels 
            labels = np.zeros(len(area_label))
            for l in range(len(area_label)):
                labels[l] = area_label[l]
            label_all.append(labels)
            feature_all.append(area_feature)
    save_dict = {}
    save_dict['feature'] = np.stack(feature_all)
    save_dict['label'] = np.stack(label_all)
    # save_dict['weight'] = np.stack(weight_all)
    utils.save_file(os.path.join(args.classifier_dir,split+'.pkl'),save_dict)
    return save_dict


def train_classifier_vaw(args, class_id):
    file = os.path.join(args.classifier_dir, 'train.pkl')
    train_data = utils.open_file(file)
    num_entries, num_classes = train_data['label'].shape 
    # create mask for -1 
    target_label = train_data['label'][:,class_id]

    mask = target_label > -1 
    updated_targets = target_label[mask]
    updated_train_data = train_data['feature'][mask]
    # change 0 to -1 
    updated_targets[updated_targets==0] = -1
    if args.balance_classes:
        class_weight = 'balanced'
    else:
        class_weight = None 
    classifier = LogisticRegression(verbose=1, multi_class='ovr', max_iter=args.iterations,class_weight=class_weight).fit(updated_train_data,
                                                                                 updated_targets,
                                                                                    sample_weight=None)
    save_dir = os.path.join(args.classifier_dir,f'model_{class_id}.sav')
    utils.save_file(save_dir,classifier)
    print(f'Saved classifier for {class_id} to {save_dir}')



def evaluate_classifier_vaw(args, class_id):
    file = os.path.join(args.classifier_dir, 'val.pkl')
    val_data = utils.open_file(file)
    loaded_model = utils.open_file(os.path.join(args.classifier_dir, f'model_{class_id}.sav'))
    target_label = val_data['label'][:,class_id]
    mask = target_label > -1 
    updated_targets = target_label[mask]
    updated_val_data = val_data['feature'][mask]
    updated_targets[updated_targets==0] = -1
    if args.use_weight:
        sample_weight = val_data['weight']
    else:
        sample_weight = None 
    try:
        roc_auc = roc_auc_score(updated_targets, loaded_model.decision_function(updated_val_data),
                                    sample_weight=sample_weight, multi_class='ovr')
    except:
        roc_auc = -1 
    try:
        ap_score = average_precision_score(updated_targets, loaded_model.decision_function(updated_val_data),
                                       sample_weight=sample_weight)
    except:
        ap_score = -1 

    print(f'ROC AUC SCORE = {roc_auc} for class {class_id}')
    print(f'Average Precision Score = {ap_score} for class {class_id}')
    return roc_auc, ap_score    



def load_features(args,split='train'):
    # assumes that the first entry in label dict is the positive class. If not use load_features_vaw
    feature_all = []
    label_all = []
    weight_all = []
    if split == 'train':
        feature_dir = args.train_region_feature_dir 
        label_dir = args.train_region_labels_dir 
    else:
        feature_dir = args.val_region_feature_dir
        label_dir = args.val_region_labels_dir 


    label_files = os.listdir(label_dir)

    for file_name in tqdm(label_files):
        if '.pkl'  not in file_name:
            file_name = file_name + '.pkl'
        file_features = utils.open_file(os.path.join(feature_dir,file_name))
        file_labels = utils.open_file(os.path.join(label_dir,file_name))

        for i, area in enumerate(file_features):
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

    clf = MLPClassifier(verbose=True,learning_rate_init=args.mlp_lr,max_iter=args.iterations,hidden_layer_sizes=[1000])
    train_data = utils.open_file(os.path.join(args.classifier_dir, 'train.pkl'))
   
    train_features = train_data['feature']
    target_label = train_data['label']
    clf.fit(train_data['feature'],train_data['label'])
    save_dir = os.path.join(args.classifier_dir,args.classifier_name,f'model_mlp.sav')
    utils.save_file(save_dir,clf)
    print(f'Saved classifier {save_dir}')
    clf = utils.open_file(os.path.join(args.classifier_dir,ars.classifier_name, f'model_mlp.sav'))
    val_data = utils.open_file(os.path.join(args.classifier_dir,'val.pkl'))
    val_features = val_data['feature']
    val_label = val_data['label']

    roc_auc = roc_auc_score(val_label, clf.predict_proba(val_features),multi_class='ovo',average='macro')
    
    ap_score = average_precision_score(val_label, clf.predict_proba(val_features),average='macro')
    print(f'ROC AUC SCORE = {roc_auc}')
    print(f'Average Precision Score = {ap_score}')
    return roc_auc, ap_score 





def evaluate_classifier(args, class_id):
    file = os.path.join(args.classifier_dir, 'val.pkl')
    val_data = utils.open_file(file)
    loaded_model = utils.open_file(os.path.join(args.classifier_dir, f'model_{class_id}.sav'))
    target_label = np.zeros_like(val_data['label'])
    target_label.fill(-1)
    target_label[np.where(val_data['label'] == class_id)] = 1
    if args.use_weight:
        sample_weight = val_data['weight']
    else:
        sample_weight = None 
    roc_auc = roc_auc_score(target_label, loaded_model.decision_function(val_data['feature']),
                                    sample_weight=sample_weight, multi_class='ovr')

    ap_score = average_precision_score(target_label, loaded_model.decision_function(val_data['feature']),
                                       sample_weight=val_data['weight'])
    print(f'ROC AUC SCORE = {roc_auc} for class {class_id}')
    print(f'Average Precision Score = {ap_score} for class {class_id}')
    return roc_auc, ap_score 


def train_classifier(args, class_id):
    file = os.path.join(args.classifier_dir, 'train.pkl')
    train_data = utils.open_file(file)
    target_label = np.zeros_like(train_data['label'])
    target_label.fill(-1)
    
    target_label[np.where(train_data['label'] == class_id)] = 1
    print('Training classifier')
    if args.use_weight:
        sample_weight = train_data['weight']
    else:
        sample_weight = None 
    
    classifier = LogisticRegression(verbose=1, multi_class='ovr', max_iter=args.iterations).fit(train_data['feature'],
                                                                                 target_label,
                                                                                    sample_weight=sample_weight)
    save_dir = os.path.join(args.classifier_dir,f'classifier/model_{class_id}.sav')
    utils.save_file(save_dir,classifier)
    print(f'Saved classifier for {class_id} to {save_dir}')


def train_and_evaluate(args):
    training_file = os.path.join(args.classifier_dir,'train.pkl')
    val_file = os.path.join(args.classifier_dir,'val.pkl')
    # root_feature, root_label, save_root,
    if not os.path.exists(training_file):
        if args.vaw:
            train_data = load_features_vaw(args,'train')
        else:
            train_data = load_features(args,'train')
    else:
        train_data = utils.open_file(training_file)

    if not os.path.exists(val_file):
        if args.vaw:
            val_data = load_features_vaw(args,'val')
        else:
            val_data = load_features(args,'val')
    else:
        val_data = utils.open_file(val_file)
    if args.start_class ==-1 or args.end_class == -1:
        # use default values
        class_ids = np.unique(train_data['label'])
        min_class = class_ids[0]
        max_class = class_ids[-1]
    else:
        min_class = args.start_class 
        max_class = args.end_class 

    if min_class == 0 and args.ignore_zero:
        min_class = 1

    avg_ap = []
    avg_roc_auc = []
    if args.mlp:
        train_mlp(args)

    else:
        for i in range(int(min_class),int(max_class)+1):
            if not args.eval_only:
                
                if args.vaw:
                    train_classifier_vaw(args,int(i))
                else:
                    train_classifier(args, int(i))
            if args.vaw:  
                roc_auc, ap_score = evaluate_classifier_vaw(args, int(i))
            else:
                roc_auc, ap_score = evaluate_classifier(args,int(i))
            avg_ap.append(ap_score)
            avg_roc_auc.append(roc_auc)
        
   
        utils.save_file(os.path.join(args.results_dir,'avg_ap.json'), avg_ap)
        utils.save_file(os.path.join(args.results_dir,'avg_roc_auc.json'), avg_roc_auc)
        print(f'Avg AP :{np.mean(avg_ap)}')
        print(f'AVG ROC AUC:{np.mean(avg_roc_auc)}')
    all_pixel_predictions, file_names = per_pixel_prediction(args)
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
        "--start_class",
        type=int,
        default=-1,
        help="Train classifier starting from this class"
    )
    parser.add_argument(
        "--end_class",
        type=int,
        default=-1,
        help="Stop classifier at this class"
    )
    parser.add_argument(
        "--use_weight",
        action="store_false",
        help="Whether to use area as weight."
    )
    parser.add_argument(
        "--ignore_unlabeled",
        action="store_false",
        help="Whether to only train regions which have a particular label. Turn off for vaw"
    )
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="To use the balance class option in scikit learn for unbalanced classes"
    )
    parser.add_argument(
        "--vaw",
        action="store_true",
        help="Multi-label option (multiple positive classes per instance)"
    )
    parser.add_argument(
        "--mlp",
        action="store_true",
        help="Use mlp"
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=.0001,
        help="Use mlp"
    )
    parser.add_argument(
        "--classifier_name",
        type=str,
        default=None,
        help="Subdirectory within classifier dir"
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
    parser.add_argument("--classifier_type",
    type=str,
    default="binary",
    help="Binary or multi-class ")
    args = parser.parse_args()
    train_and_evaluate(args)