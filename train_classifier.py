from sklearn.linear_model import LogisticRegression
import pickle
import os
import numpy as np
from tqdm import tqdm
import argparse 
import gc
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
def load_features(root_feature, root_label,save_root, split='training'):
    feature_all = []
    label_all = []
    weight_all = []

    label_root = os.path.join(root_label, split)
    label_files = os.listdir(label_root)

    for file_name in tqdm(label_files):
        with open(os.path.join(root_feature, file_name), 'rb') as f:
            file_features = pickle.load(f)

        with open(os.path.join(label_root, file_name), 'rb') as f:
            file_labels = pickle.load(f)

        for i, area in enumerate(file_features):
            area_feature = area['pooled_region']
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

    with open(os.path.join(save_root, split + '.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)
    return save_dict


def evaluate_classifier(root_feature, root_label, save_root,class_id):
    file = os.path.join(save_root, 'validation.pkl')
    if os.path.exists(file):
        with open(file, 'rb') as f:
            print('Loading saved val data')
            val_data = pickle.load(f)
    else:
        print('Combining val features')
        val_data = load_features(root_feature, root_label, save_root,'validation')

 
    # load the model from disk
    filename = f'{save_root}/model_{class_id}.sav'
    with open(filename,'rb') as g:
        loaded_model = pickle.load(g)

    target_label = np.zeros_like(val_data['label'])
    target_label.fill(-1)
    target_label[np.where(val_data['label'] == class_id)] = 1
    roc_auc_score = roc_auc_score(target_label,loaded_model.decision_function(val_data['feature']),sample_weight=val_data['weight'], multi_class='ovr')
    ap_score = average_precision_score(target_label,loaded_model.decision_function(val_data['feature']),sample_weight=val_data['weight'])
    print(f'ROC AUC SCORE = {roc_auc_score} for class {class_id}')
    print(f'Average Precision Score = {ap_score} for class {class_id}')
    del loaded_model


def train_classifier(root_feature, root_label, save_root, class_id):
    file = os.path.join(save_root, 'training.pkl')
    if os.path.exists(file):
        with open(file, 'rb') as f:
            print('Loading saved features')
            train_data = pickle.load(f)
    else:
        print('Combining training features')
        train_data = load_features(root_feature, root_label,save_root, 'training')

    print(f'Learning features for {class_id}')
    target_label = np.zeros_like(train_data['label'])
    target_label.fill(-1)
    target_label[np.where(train_data['label'] == class_id)] = 1
    print('Training classifier')
    classifier = LogisticRegression(verbose=1, multi_class='ovr', max_iter=200).fit(train_data['feature'],
                                                                                    target_label,
                                                                                    sample_weight=train_data['weight'])
    # save the model to disk
    filename = f'{save_root}/model_{class_id}.sav'
    with open(filename,'wb') as s:
        pickle.dump(classifier, s)
    del classifier



if __name__ == '__main__':
    root_feature = '/shared/rsaas/dino_sam/pooled_features_pkl'
    root_label = '/shared/rsaas/dino_sam/labels/ADE20K'
    save_root = '/shared/rsaas/dino_sam'


    parser = argparse.ArgumentParser()

    # get class ids for training 
    parser.add_argument('--max_class_id',default=-1)
    parser.add_argument('--min_class_id',default=-1)
    parser.add_argument('--include_0_class',action='store_true')
    args = parser.parse_args()
    training_file = os.path.join(save_root, 'training.pkl')
    val_file = os.path.join(save_root,'validation.pkl')
    if not os.path.exists(training_file):
        train_data = load_features(root_feature, root_label,save_root, 'training')
    if not os.path.exists(val_file):
        val_data = load_features(root_feature, root_label,save_root, 'validation')

    
    if args.max_class_id == -1 or args.min_class_id == -1:
        class_ids = np.unique(train_data['labels'])
        min_class = class_ids[0]
        max_class= class_ids[-1]

        if class_ids[0] == 0 and args.include_0_class == False:
            # we do not want to train the 0th class 
            min_class  = class_ids[1]
    else:
        min_class = args.min_class_id
        max_class = args.max_class_id

  
    for i in range(int(min_class),int(max_class)+1):

        train_classifier(root_feature, root_label, save_root,int(i))
        evaluate_classifier(root_feature, root_label, save_root,int(i))
   