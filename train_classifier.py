from sklearn.linear_model import LogisticRegression
import pickle
import os
import numpy as np
from tqdm import tqdm


def load_features(root_feature, root_label, split='training'):
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
        pickle.dump(save_dict, f, protocol=4)
    return save_dict


def evaluate_classifier(root_feature, root_label, save_root):
    file = os.path.join(save_root, 'validation.pkl')
    if os.path.exists(file):
        with open(file, 'rb') as f:
            val_data = pickle.load(f)
    else:
        val_data = load_features(root_feature, root_label, 'validation')

    for class_id in np.unique(val_data['label']):
        # load the model from disk
        filename = f'{save_root}/model_{class_id}.sav'
        loaded_model = pickle.load(open(filename, 'rb'))

        target_label = np.zeros_like(val_data['label'])
        target_label.fill(-1)
        target_label[np.where(val_data['label'] == class_id)] = 1

        acc = loaded_model.score(val_data['feature'], target_label, sample_weight=val_data['weight'])
        print(f'ACC for Class {class_id}: {acc}')


def train_classifier(root_feature, root_label, save_root):
    file = os.path.join(save_root, 'training.pkl')
    if os.path.exists(file):
        with open(file, 'rb') as f:
            train_data = pickle.load(f)
    else:
        train_data = load_features(root_feature, root_label, 'training')

    for class_id in np.unique(train_data['label']):
        target_label = np.zeros_like(train_data['label'])
        target_label.fill(-1)
        target_label[np.where(train_data['label'] == class_id)] = 1
        
        classifier = LogisticRegression(verbose=1, multi_class='ovr', max_iter=200).fit(train_data['feature'],
                                                                                        target_label,
                                                                                        sample_weight=train_data['weight'])
        # save the model to disk
        filename = f'{save_root}/model_{class_id}.sav'
        pickle.dump(classifier, open(filename, 'wb'))



if __name__ == '__main__':
    root_feature = '/shareData/DINO_SAM/pooled_features_pkl'
    root_label = '/shareData/DINO_SAM/labels/ADE20K'
    save_root = '/shareData/DINO_SAM'

    train_classifier(root_feature, root_label, save_root)
    evaluate_classifier(root_feature, root_label, save_root)