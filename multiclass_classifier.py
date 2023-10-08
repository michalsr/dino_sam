import gc
import os
import tqdm
import utils
import torch
import pickle
import argparse 
import numpy as np

from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
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

class CustomDataset(Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        self.num_classes = torch.unique(label).shape[0]
        
    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.label[idx]

class CustomLogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomLogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs
    
def custom_loss(pred, target):
    # pred_ignore_zero = pred[:, 1:]
    exp_pred = torch.exp(pred)
    target_ignore_zero = target-1
    
    loss = 0
    for i in range(target.shape[0]):
        if target[i].item() != 0:
            loss += -torch.log(exp_pred[i][target_ignore_zero[i]]/torch.sum(exp_pred[i])+1)
            # loss += -torch.log(1/(1+torch.sum(exp_pred[i][exp_pred[i]!=exp_pred[i][target_ignore_zero[i]]])))
        else:
            loss += -torch.log(1/(1+torch.sum(exp_pred[i])))
    return loss

    
def train_eval_classifier(args):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    
    file = os.path.join(args.classifier_dir, 'train.pkl')
    train_data = utils.open_file(file)
    target_label = train_data['label']
    train_feature = torch.from_numpy(train_data['feature']).to(device)
    train_label = torch.from_numpy(train_data['label']).to(device)
    
    file = os.path.join(args.classifier_dir, 'val.pkl')
    val_data = utils.open_file(file)
    target_label = val_data['label']
    val_feature = torch.from_numpy(val_data['feature']).to(device)

    val_label = torch.from_numpy(val_data['label']).to(device)
    
    input_dim = train_feature.shape[1]
    if args.with_background_class:
        output_dim = np.unique(train_data['label']).shape[0]-1
    else:
        output_dim = np.unique(train_data['label']).shape[0]
    
    model = CustomLogisticRegression(input_dim=input_dim, output_dim=output_dim)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    epochs = args.epoch
    batch_size = args.batch_size
    
    train_loss = []
    val_loss = []
    train_dataset = CustomDataset(train_feature, train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = CustomDataset(val_feature, val_label)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in tqdm(range(epochs)):
        loss1, loss2 = 0, 0
        for i, (features, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(features)
            loss_train = custom_loss(outputs, labels)
            loss_train.backward()
            optimizer.step()
            loss1 += loss_train.item()
        
        train_loss.append(loss1/len(train_loader))
        
        for i, (features, labels) in enumerate(val_loader):
            outputs = model(features)
            loss_eval = custom_loss(outputs, labels)
            loss2 += loss_eval.item()
            
        val_loss.append(loss2/len(val_loader))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_train,
            }, os.path.join(args.classifier_dir, 'model.pt'))
    
    return model, val_loader

def calculate_accuracy(model, loader, size):
    correct = 0
    model.eval()
    for X, y in loader:
        output = model(X)
        output_exp = torch.exp(output)
        exp_sum = torch.sum(output_exp, axis=1)
        pred = torch.div(output_exp.T, (exp_sum+1)).T
        pred_max, pred_indices = torch.max(output, dim=1)
        other = 1/(exp_sum+1)
        pred_final = torch.where(pred_max>other, pred_indices+1, 0) 
        equal = torch.sum(torch.eq(y, pred_final))
        correct += equal
    return correct/size



def train_and_evaluate_other(args):
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    
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
        
    if not args.eval_only:
        model, val_loader = train_eval_classifier(args)
        acc = calculate_accuracy(model, val_loader, val_data['feature'].shape[0]).item()
        utils.save_file(os.path.join(args.results_dir,'acc.json'), acc)
    return acc

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
    parser.add_argument(
        "--with_background_class",
        action="store_true",
        help="Dataset contains background or not"
    )
    parser.add_argument(
            "--epoch",
            type = int,
            default=200,
            help="Number of training epoch"
        )
    parser.add_argument(
            "--batch_size",
            type = int,
            default=32,
            help="Batch size"
        )
    parser.add_argument(
            "--learning_rate",
            type = int,
            default=0.001,
            help="Batch size"
        )
    args, unknown = parser.parse_known_args()
    # train_and_evaluate(args)
    train_and_evaluate_other(args)
