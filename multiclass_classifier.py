import gc
import os
import utils 
import pickle
import argparse 
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
"""
Train and eval multiclass classifiers with 'other' label
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

def evaluate_classifier(args):
    file = os.path.join(args.classifier_dir, 'val.pkl')
    val_data = utils.open_file(file)
    target_label = val_data['label']
    loaded_model = utils.open_file(os.path.join(args.classifier_dir, f'model_multiclass.sav'))
    
    acc = loaded_model.score(val_data['feature'], target_label, sample_weight=val_data['weight'])
    
    print(f'ACC for multiclass: {acc}')
    return acc

def train_classifier(args):
    file = os.path.join(args.classifier_dir, 'train.pkl')
    train_data = utils.open_file(file)
    target_label = train_data['label']
    classifier = LogisticRegression(verbose=1, multi_class='multinomial', max_iter=args.iterations).fit(train_data['feature'],
                                                                                 target_label,
                                                                                    sample_weight=train_data['weight'])
    save_dir = os.path.join(args.classifier_dir,f'model_multiclass.sav')
    utils.save_file(save_dir,classifier)
    print(f'Saved multiclass clssifier')

def train_and_evaluate(args):
    print(args.classifier_dir)
    training_file = os.path.join(args.classifier_dir,'train.pkl')
    val_file = os.path.join(args.classifier_dir,'val.pkl')
    # root_feature, root_label, save_root,
    if not os.path.exists(training_file):
        train_data = load_features(args,'train')
    else:
        train_data = utils.open_file(training_file)
    
    if not os.path.exists(val_file):
        val_data = load_features(args,'val')
        
    if not args.eval_only:
        train_classifier(args)
    
    acc = evaluate_classifier(args)
    utils.save_file(os.path.join(args.results_dir,'acc.json'), acc)

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
        help="Location to store accuracy result"
    )
    parser.add_argument(
            "--eval_only",
            action="store_true",
            help="No classifier training"
        )
    args, unknown = parser.parse_known_args()
    train_and_evaluate(args)
