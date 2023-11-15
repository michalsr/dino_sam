import sys
import pickle
import json
from typing import List
from pycocotools import mask as mask_utils
# from einops import rearrange, reduce
import torch
import einops
import numpy as np
import torch.nn.functional as F
from utils import mean_iou
from PIL import Image

import itertools
import math
import argparse
from tqdm import tqdm
from torch import nn, optim
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import utils
import torchvision

import sys
import pickle
import json
from typing import List
from pycocotools import mask as mask_utils
# from einops import rearrange, reduce
import torch
import einops
import numpy as np
import torch.nn.functional as F
from utils import mean_iou
from PIL import Image

import itertools
import math
import argparse
from tqdm import tqdm
from torch import nn, optim
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import utils
import torchvision
'''
Use transformer to classify regions
'''

def get_all_features(region_feat_dir, region_labels_dir,pos_embd_dir,data_file):
    if os.path.exists(data_file):
        data = utils.open_file(data_file)
        return np.stack(data['features']),np.stack(data['labels']),np.stack(data['weight'])

    all_feats = []
    all_labels = []
    all_weight = []
    print('Loading features')
    for file_name in tqdm(os.listdir(region_feat_dir)):
        region_feats = utils.open_file(os.path.join(region_feat_dir,file_name))
        labels = utils.open_file(os.path.join(region_labels_dir,file_name))

        if pos_embd_dir is not None:
            pos_embd = utils.open_file(os.path.join(pos_embd_dir,file_name))

        for i,region in enumerate(region_feats):
            if pos_embd_dir is None:
                area_feature = region['region_feature']
            else:
                area_feature = region['region_feature']+pos_embd[i,:]

            area_label = labels[i]['labels']
            area_weight = region['area']
            target_label = list(area_label.keys())[0]

            if area_label[target_label] == 1:
                if target_label == 0:
                    break
                else:
                    all_feats.append(area_feature)
      
                    all_labels.append(target_label)
                    all_weight.append(area_weight)

    utils.save_file(data_file,{'features':all_feats,'labels':all_labels,'weight':all_weight})
    return np.stack(all_feats), np.stack(all_labels),np.stack(all_weight)



class FeatureDataset(Dataset):
    # load region features, add positional encoding and get region labels
    def __init__(self,region_feat_dir, region_labels_dir,pos_embd_dir,data_file):
        super().__init__()
        region_feats,region_labels,weight = get_all_features(region_feat_dir, region_labels_dir,pos_embd_dir,data_file)
        self.region_feats = region_feats
        self.labels = region_labels
        self.weight = weight


    def __len__(self):
        return len(self.region_feats)

    def __getitem__(self, idx):
        region_feats = self.region_feats[idx]
        labels = self.labels[idx]
        weight = self.weight[idx]
        return torch.tensor(region_feats), torch.tensor(labels), torch.tensor(weight)
def eval_acc(args,model,epoch):
    dataset = FeatureDataset(region_feat_dir=args.val_region_feature_dir,region_labels_dir=args.val_region_labels_dir,pos_embd_dir=args.val_pos_embd_dir,data_file=args.val_data_file)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    
    if args.ade:
        criterion = nn.CrossEntropyLoss(reduction='sum',ignore_index=0)
    else:
        criterion = nn.CrossEntropyLoss(reduction='sum')
    mca = MulticlassAccuracy(num_classes=args.num_classes+1, average='micro',top_k=1)
    predictions = []
    all_labels = []
    total_regions = 0
    all_loss = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            region_feats, labels, weight= data
            total_regions += len(labels)
            model = model.cuda()

            labels = labels.cuda()
            region_feats = region_feats.cuda()
            outputs = model(region_feats)

            outputs = outputs.squeeze()


            # Reshape outputs and labels for loss calculation
            outputs = outputs.view(-1, args.num_classes+1)
            predictions.append(outputs.cpu())
            # print(outputs.shape)
            labels = labels.view(-1)
            all_labels.append(labels.cpu())

            loss = criterion(outputs, labels)
            all_loss+=(loss.item())


    val_loss = all_loss/total_regions
    print(f'Val loss:{val_loss}')
    predictions = torch.stack(predictions)
    all_labels = torch.stack(all_labels)
    # print(predictions.shape,all_labels.shape)
    val_acc = mca(predictions.squeeze(),all_labels.squeeze())
    print(f'Val acc:{val_acc.item()}')
    return val_loss,val_acc.item()


def train_model(args):
    dataset = FeatureDataset(region_feat_dir=args.train_region_feature_dir,region_labels_dir=args.train_region_labels_dir,pos_embd_dir=args.train_pos_embd_dir,data_file=args.train_data_file)
    if args.model == 'linear':
        model = torch.nn.Linear(args.input_channels,args.num_classes+1)
    else:
        model = torchvision.ops.MLP(in_channels=args.input_channels,hidden_channels=[args.hidden_channels,args.num_classes+1])

    eval_acc(args,model,1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    if args.ade:
        criterion = nn.CrossEntropyLoss(reduction='none',ignore_index=0)
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')
    epochs = args.epochs
    mca = MulticlassAccuracy(num_classes=args.num_classes+1, average='micro',top_k=1)
    # batch is over total number of regions so can make it very large (8192)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    print(f'Train dataloader length with batch size {args.batch_size}: {len(dataloader)}')
    train_outputs = []
    train_labels = []
    total_regions = 0
    for epoch in range(epochs):  # Example number of epochs
        batch_loss = 0
        model = model.cuda()
        model.train()
        num_regions = 0
        train_acc = 0
        for i, data in enumerate(tqdm(dataloader)):
            model.train()
            region_feats, labels,weight = data
            num_regions += len(labels)
            region_feats = region_feats.cuda()
            labels = labels.cuda()

            outputs = model(region_feats)
            outputs = outputs.squeeze()

            outputs = outputs.view(-1, args.num_classes+1)

            labels = labels.view(-1)
            weight = weight.cuda()
            weight = torch.nn.functional.normalize(weight.float(),dim=0)


            loss = (criterion(outputs, labels)*weight).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
       
                train_acc += (mca(outputs.cpu(),labels.cpu()).item() * labels.size()[0])



        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(model.cpu().state_dict(),os.path.join(args.save_dir,'model.pt'))
        val_loss,val_acc = eval_acc(args,model,epoch)
        train_acc = train_acc/num_regions
        print(f"Train_acc:{train_acc}")
        metrics = {'val_loss':val_loss,'val_acc':val_acc,'train_acc':train_acc,'train_loss':loss.item()}
        utils.save_file(os.path.join(args.results_dir,f'metrics_epoch_{epoch}.json'),metrics,json_numpy=True)
        scheduler.step()

        if (epoch+1)%args.iou_every==0:
            all_pixel_predictions, file_names = eval_model(args)
            compute_iou(args,all_pixel_predictions,file_names,epoch)


def eval_model(args):
    if args.model == 'linear':
        model = torch.nn.Linear(args.input_channels, args.num_classes+1)
    else:
        model = torchvision.ops.MLP(in_channels=args.input_channels,hidden_channels=[args.hidden_channels,args.num_classes+1])
    
    model.load_state_dict(torch.load(os.path.join(args.save_dir,'model.pt')))
    model.eval()
    class_preds = []
    model = model.cuda()
    all_pixel_predictions = []
    # keep track of order of predictions
    file_names = []
    val_features = args.val_region_feature_dir
    print('val features dir: ', val_features)
    softmax = torch.nn.Softmax(dim=1)
    val_files = [filename for filename in os.listdir(val_features)]
    for file in tqdm(val_files):
        file_names.append(file)
        all_sam = utils.open_file(os.path.join(args.sam_dir,file.replace('.pkl','.json')))

        if args.val_pos_embd_dir is not None:
            pos_embd = utils.open_file(os.path.join(args.val_pos_embd_dir,file))

        all_regions = []
        region_order = []
        region_features = utils.open_file(os.path.join(val_features,file))
        feature_all = []
        for j,area in enumerate(region_features):
            if args.val_pos_embd_dir is None:
                features = area['region_feature']
            else:
                features = area['region_feature'] + pos_embd[i,:]

            feature_all.append(features)

        region_all = {area['region_id']:j for j,area in enumerate(region_features)}
        # track region id
        region_idx = []
        for i, region in enumerate(all_sam):
            if region['region_id'] not in region_all.keys():
                continue
            else:
                region_idx.append(region_all[region['region_id']])
                region_order.append(region['region_id'])
                mask = mask_utils.decode(region['segmentation'])
                all_regions.append(mask.astype('float32'))

        if len(feature_all) == 0: # There were no predicted regions; use None as a flag
            all_pixel_predictions.append(None)
            continue

        features = torch.tensor(np.stack(feature_all))
        features = features[region_idx,:]

        predictions = torch.zeros((len(feature_all),args.num_classes+1))
        with torch.no_grad():
            for i in range(len(feature_all)):
                feats = features[i,:]

                model = model.cuda()

                feats = feats.cuda().unsqueeze(0)

                output = model(feats)
                predictions[i,:] = output.cpu()

        if 'after_softmax' in args.multi_region_pixels:
            # averaging softmax values for pixels in multiple regions
            class_predictions = softmax(predictions)
        else:
            # use logits for predictions
            class_predictions = predictions

        num_regions, num_classes = class_predictions.size()

        all_regions = torch.from_numpy(np.stack(all_regions,axis=-1))
        class_predictions = class_predictions.cuda()
        h,w,num_regions = all_regions.size()

        #find pixels where at least one mask equals one
        mask_sum = torch.sum(all_regions,dim=-1)

        mask_sum = mask_sum.cuda()
        nonzero_mask = torch.nonzero(mask_sum,as_tuple=True)

        all_regions = all_regions.cuda()

        nonzero_regions = all_regions[nonzero_mask[0],nonzero_mask[1],:]
        product = torch.matmul(nonzero_regions, class_predictions)

        # want avg across softmax values, need to get number of regions summed for each pixel
        # repeat number of regions across softmax values

        divide = torch.repeat_interleave(mask_sum[nonzero_mask[0],nonzero_mask[1],None],num_classes,dim=1)

        nonzero_region_pixel_preds = torch.divide(product,divide)

        if 'before_softmax' in args.multi_region_pixels:
            nonzero_region_pixel_preds = softmax(nonzero_region_pixel_preds,dim=1)
        top_pred = torch.argmax(nonzero_region_pixel_preds,dim=1).cpu().numpy()
        final_pixel_pred = np.zeros((h,w))

        # index back into original shape
        final_pixel_pred[nonzero_mask[0].cpu().numpy(),nonzero_mask[1].cpu().numpy()] = top_pred
        all_pixel_predictions.append(final_pixel_pred)
    return all_pixel_predictions, file_names

def compute_iou(args,predictions,file_names,epoch):
    actual_labels = []
    for file in tqdm(file_names):
        actual = np.array(Image.open(os.path.join(args.annotation_dir,file.replace('.pkl','.png'))))
        actual_labels.append(actual)

    # Handle predictions where there were no regions
    predictions = [np.full(actual.shape, 255) if p is None else p for p, actual in zip(predictions, actual_labels)]

    if args.ignore_zero or args.ade:
        num_classes = args.num_classes-1
        reduce_labels = True
        reduce_pred_labels=True 

    else:
        num_classes = args.num_classes+1
        reduce_labels = False
        reduce_pred_labels=False 
    if args.ade==True:
        assert reduce_labels==True 
        assert reduce_pred_labels==True 
        assert num_classes == 149
    miou = mean_iou(results=predictions,gt_seg_maps=actual_labels,num_labels=num_classes,ignore_index=255,reduce_labels=reduce_labels,reduce_pred_labels=reduce_pred_labels)
    print(miou)
    utils.save_file(os.path.join(args.results_dir,f'mean_iou_epoch_{epoch}.json'),miou,json_numpy=True)

def train_and_evaluate(args):
    if args.num_classes != 150 and args.num_classes!= 20:
        raise ValueError('ADE should have 150 and Pascal VOC should have 20. The background class is taken care of in the code')
    if args.num_classes == 150:
        if args.ade ==False:
            raise ValueError('If using ADE then ade argument should be set to True')
    if args.ade==True:
        print('Training and evaluating on ADE. Make sure to use the correct region label directory (ADE20K_no_zero)!')
    if not args.eval_only:
        train_model(args)

    all_pixel_predictions, file_names = eval_model(args)
    compute_iou(args,all_pixel_predictions,file_names,args.epochs)
    # Save pixel predictions as PNGs for use on evaluation server
    if args.output_predictions:
        print('Saving predictions to PNGs')
        prediction_dir = os.path.join(args.results_dir, 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)

        for file_name, prediction in tqdm(zip(file_names, all_pixel_predictions)):
            prediction = Image.fromarray(prediction.astype(np.uint8))
            prediction.save(os.path.join(prediction_dir, file_name.replace('.pkl', '.png')))

    else: # No need to output predictions if evaluating here
        compute_iou(args,all_pixel_predictions,file_names,args.epochs)

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
    parser.add_argument('--epochs',
                        type=int,
                         default=2,
                         help='Number of iterations to run log regression')
    parser.add_argument(
        "--ignore_zero",
        action="store_true",
        help="Include 0 class"
    )
    parser.add_argument(
        "--train_data_file",
        type=str,
        default=None,
        help="Location of region data."
    )
    parser.add_argument(
        "--val_data_file",
        type=str,
        default=None,
        help="Location of region data. Created if None"
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
        "--save_dir",
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
    parser.add_argument("--dataset_name",type=str,default='ade')

    parser.add_argument(
        "--lr",
        type=float,
        default=.0001,
        help="learning rate"
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
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        help="linear or mlp")

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
        "--batch_size",
        type=int,
        default=8192,
        help="Batch"
    )
    parser.add_argument(
        "--iou_every",
        type=int,
        default=1,
        help="Compute iou every"
    )
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=512,
        help="hidden channel size if used"
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=1024,
gs        help="input channel size depending on models"
    )
    parser.add_argument(
        '--output_predictions',
        action='store_true',
        help='Output predictions as PNGs'
    )
    parser.add_argument(
        '--ade',
        action='store_true',
        help='Output predictions as PNGs'
    )

    args = parser.parse_args()
    train_and_evaluate(args)
