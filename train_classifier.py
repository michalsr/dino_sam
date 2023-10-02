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
"""
Train and eval binary classifiers 
"""

def load_features(args,split='train'):
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


def evaluate_classifier(args, class_id):
    file = os.path.join(args.classifier_dir, 'val.pkl')
    val_data = utils.open_file(file)
    loaded_model = utils.open_file(os.path.join(args.classifier_dir,f'model_{class_id}.sav'))
    target_label = np.zeros_like(val_data['label'])
    target_label.fill(-1)
    target_label[np.where(val_data['label'] == class_id)] = 1
    roc_auc = roc_auc_score(target_label, loaded_model.decision_function(val_data['feature']),
                                  sample_weight=val_data['weight'], multi_class='ovr')
    ap_score = average_precision_score(target_label, loaded_model.decision_function(val_data['feature']),
                                       sample_weight=val_data['weight'])
    print(f'ROC AUC SCORE = {roc_auc_score} for class {class_id}')
    print(f'Average Precision Score = {ap_score} for class {class_id}')
    return roc_auc, ap_score 


def train_classifier(args, class_id):
    file = os.path.join(args.classifier_dir, 'train.pkl')
    train_data = utils.open_file(file)
    if class_id != 0:
        train_data['label'] = train_data['label'][train_data['label']!=0]
        train_data['feature'] = train_data['feature'][:,train_data['label']!=0]
        train_data['weight'] = train_data['weight'][train_data['label']!=0]
    target_label = np.zeros_like(train_data['label'])
    target_label.fill(-1)
    
    target_label[np.where(train_data['label'] == class_id)] = 1
    print('Training classifier')
    classifier = LogisticRegression(verbose=1, multi_class='ovr', max_iter=args.iterations).fit(train_data['feature'],
                                                                                 target_label,
                                                                                    sample_weight=train_data['weight'])
    save_dir = os.path.join(args.classifier_dir,f'model_{class_id}.sav')
    utils.save_file(save_dir,classifier)
    print(f'Saved classifier for {class_id} to {save_dir}')





def train_and_evaluate(args):

    result_dir = args.result_dir
    training_file = os.path.join(args.classifier_dir,'train.pkl')
    val_file = os.path.join(args.classifier_dir,'val.pkl')
    # root_feature, root_label, save_root,
    if not os.path.exists(training_file):
        train_data = load_features(args,'train')
    else:
        train_data = utils.open_file(training_file)
    if not os.path.exists(val_file):
        val_data = load_features(args,'val')
    else:
        val_data = utils.open_file(val_file)
    class_ids = np.unique(train_data['label'])
    min_class = class_ids[0]
    max_class = class_ids[-1]
    if class_ids[0] and args.ignore_zero:
        min_class = class_ids[1]
    avg_ap = []
    avg_roc_auc = []
    for i in range(int(min_class),int(max_class)+1):
        if not args.eval_only:
            train_classifier(args, int(i))
        roc_auc, ap_score  = evaluate_classifier(args,int(i))
        avg_ap.append(ap_score)
        avg_roc_auc.append(roc_auc)
   
    utils.save_json_file(os.path.join(args.results_dir,'avg_ap.json'),avg_ap)
    utils.save_json_file(os.path.join(args.results_dir,'avg_roc_auc.json'),avg_roc_auc)
    print(f'Avg AP :{np.mean(avg_ap)}')
    print(f'AVG ROC AUC:{np.mean(avg_roc_auc)}')

    
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
                         default=200,
                         help='Number of iterations to run log regression')
    parser.add_argument(
        "--ignore_zero",
        action="store_true",
        help="Include 0 class"
    )
    parser.add_argument(
        "--num_classes",
        default=0,
        help="Number of classes in dataset"
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
        "--result_dir",
        type=str,
        default=None,
        help="Location to store AP, AUC-ROC results"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="No classifier training"
    )
    args = parser.parse_args()
    train_and_evaluate(args)

    