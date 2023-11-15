import sys
import pickle
import json
from typing import List
from pycocotools import mask as mask_utils
# from einops import rearrange, reduce
import torch
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from utils import mean_iou
from PIL import Image
from scipy.special import softmax, logit
import itertools
import math
import argparse
from tqdm import tqdm
from torch import nn, optim
from torch.optim import Adam
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import utils
class RegionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes):
        super(RegionTransformer, self).__init__()

        # Multi-Head Attention
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,nhead=num_heads,activation='gelu',batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # Final linear layer to output area-wise labels
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        heads_layers["pre_logits"] = nn.Linear(embed_dim, embed_dim//2)
        heads_layers["act"] = nn.Tanh()
        heads_layers["head"] = nn.Linear(embed_dim//2, num_classes)
        self.heads = nn.Sequential(heads_layers)
        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, region_feats):
        # q,k,v are all dino features averaged within sam region+positional encoding
        #region_feats = region_feats.unsqueeze(0)

        output = self.encoder(region_feats)

        # Predict area-wise labels
        output = self.heads(output)


        return output
class FeatureDataset(Dataset):
    # load region features, add positional encoding and get region labels
    def __init__(self,region_feat_dir, region_labels_dir,pos_embd_dir,args):
        super().__init__()
        #region_feats,region_labels = get_all_features(region_feat_dir, region_labels_dir,pos_embd_dir)
        self.region_feats = [f for f in os.listdir(region_feat_dir) if f.endswith('.pkl')]
        self.region_labels_dir = region_labels_dir
        self.pos_embd_dir = pos_embd_dir
        self.region_feat_dir = region_feat_dir
        self.args = args

    def __len__(self):
        return len(self.region_feats)

    def __getitem__(self, idx):
        file_name= self.region_feats[idx]

        region_feats = utils.open_file(os.path.join(self.region_feat_dir,file_name))
        labels = utils.open_file(os.path.join(self.region_labels_dir,file_name))
        pos_embd = utils.open_file(os.path.join(self.pos_embd_dir,file_name))

        # combine and stack
        all_feats = []
        all_labels = []
        all_weight = []
        for i,region in enumerate(region_feats):
            area_feature = region['region_feature']+pos_embd[i,:]
            area_label = labels[i]['labels']
            area_weight = region['area']
            target_label = list(area_label.keys())[0]

            if area_label[target_label] == 1:
                all_feats.append(area_feature)
                all_weight.append(area_weight)

       
        if len(all_feats) == 0:
            all_feats = np.zeros(1)
            all_labels = np.zeros(1)
            all_weight = np.zeros(1)
            flag = True
        else:
            all_feats = np.stack(all_feats)
            all_labels = np.stack(all_labels)
            all_weight = np.stack(all_weight)
            flag = False

        return torch.tensor(all_feats), torch.tensor(all_labels),torch.tensor(all_weight),flag
def eval_acc(args,model):
    dataset = FeatureDataset(region_feat_dir=args.val_region_feature_dir,region_labels_dir=args.val_region_labels_dir,pos_embd_dir=args.val_pos_embd_dir,args=args)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    mca = MulticlassAccuracy(num_classes=args.num_classes, average='micro',top_k=1)
    batch_acc = 0
    predictions = []
    all_labels = []
    total_regions = 0
    all_loss = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader)):
            region_feats, labels,_,_ = data
            
            model = model.cuda()

            labels = labels.cuda()
            region_feats = region_feats.cuda()
            outputs = model(region_feats)
            # print(outputs.size())
            outputs = outputs.squeeze(0)
            #outputs = outputs.transpose(0, 1).view(-1, args.num_classes)
            labels = labels.view(-1)
            total_regions += labels.size()[0]

            loss = criterion(outputs, labels)
            all_loss+=(loss.item())

            batch_acc += (mca(outputs.cpu(),labels.cpu()).item() * total_regions)
    val_loss = all_loss/total_regions
    print(f'Val loss:{val_loss}')
    batch_acc = batch_acc/total_regions
    print(f'Batch acc:{batch_acc}')

    return val_loss,batch_acc




def train_transformer(args):
    dataset = FeatureDataset(region_feat_dir=args.train_region_feature_dir,region_labels_dir=args.train_region_labels_dir,pos_embd_dir=args.train_pos_embd_dir,args=args)
    model = RegionTransformer(1024,8,args.num_classes+1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    if args.use_weight:
        reduction='none'
    else:
        reduction='mean'
    if args.ade:
        ignore_index = 0
    else:
        ignore_index = -1000
    criterion = nn.CrossEntropyLoss(reduction=reduction,ignore_index=ignore_index) # ANSEL Allow mean reduction to normalize loss by number of examples (regions) in a batch
 
    epochs = args.epochs

    # ANSEL batch size will be the number of images in a batch, not regions
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True)

    mca = MulticlassAccuracy(num_classes=args.num_classes, average='micro',top_k=1)
    for epoch in range(epochs):  # Example number of epochs
        model = model.cuda()
        for i, data in enumerate(tqdm(dataloader), start=1):
            model.train()
            region_feats, labels,weights,flag = data
            if flag:
                continue

            region_feats = region_feats.cuda()
            labels = labels.cuda()
            weights = weights.cuda()

            outputs = model(region_feats)

            # ANSEL assuming outputs are of shape (bsize, n_regions, n_classes)
            outputs = outputs.reshape(-1, outputs.shape[-1]) # (bsize * n_regions, n_classes)
            labels = labels.reshape(-1) # (bsize * n_regions)

            # ANSEL Update region train accuracy
            mca(outputs.detach(), labels) # ANSEL No need to shift to CPU

            # ANSEL Compute loss and take optimizer step every accumulate_grad_batches batches.
            # Effective image batch size is then args.batch_size * args.accumulate_grad_batches
            if args.use_weight:
                weight = torch.nn.functional.normalize(weight.float(),dim=0)
                loss = (citerion(outputs,labels)*weights.cuda()).mean()
            else:
                loss = criterion(outputs,labels)
            loss = criterion(outputs, labels)
            loss = loss / args.accumulate_grad_batches # ANSEL accumulate_grad_batches is 1 by default
            loss.backward()

            if i % args.accumulate_grad_batches == 0 or i == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
        torch.save(model.cpu().state_dict(),os.path.join(args.save_dir,'model.pt'))
        val_loss,val_acc = eval_acc(args,model)

        train_acc = mca.compute()
        print(f"Train_acc:{train_acc}")
        metrics = {'val_loss':val_loss,'val_acc':val_acc,'train_acc':train_acc,'train_loss':loss.item()}
        utils.save_file(os.path.join(args.results_dir,f'metrics_epoch_{epoch}.json'),metrics,json_numpy=True)

        if (epoch+1)%args.iou_every==0:
            all_pixel_predictions, file_names = eval_transformer(args)
            compute_iou(args,all_pixel_predictions,file_names)
        scheduler.step()


def eval_transformer(args):
    model = RegionTransformer(1024,8,args.num_classes+1)
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
        for i, region in enumerate(all_sam):
            mask = mask_utils.decode(region['segmentation'])
            all_regions.append(mask.astype('float32'))
            region_order.append(region['region_id'])
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
        region_idx = [region_all[r] for r in region_order]

        if len(feature_all) == 0: # There were no predicted regions; use None as a flag
            all_pixel_predictions.append(None)
            continue

        features = torch.tensor(np.stack(feature_all))
        features = features[region_idx,:]

        
        with torch.no_grad():
            feats = features 
            feats = feats.cuda()
            model=  model.cuda()
            output = model(feats)
            predictions = output.cpu()
            

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

def compute_iou(args,predictions,file_names):
    actual_labels = []
    for file in tqdm(file_names):
        actual = np.array(Image.open(os.path.join(args.annotation_dir,file.replace('.pkl','.png'))))
        actual_labels.append(actual)
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

def train_and_evaluate(args):
    if args.num_classes != 150 and args.num_classes!= 20:
        raise ValueError('ADE should have 150 and Pascal VOC should have 21')
    if args.num_classes == 150:
        if args.ade ==False:
            raise ValueError('If using ADE then ade argument should be set to True')
    if args.ade==True:
        print('Training and evaluating on ADE. Make sure to use the correct region label directory (ADE20K_no_zero)!')
    if not args.eval_only:
        train_transformer(args)
    all_pixel_predictions, file_names = eval_transformer(args)
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
        "--dataset_name",
        type=str,
        default='ade',
        help="If ade or not"
    )
    parser.add_argument(
        "--iou_every",
        type=int,
        default=10,
        help="when to compute iou"
    )

    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to perform gradient accumulation over."
    )
    parser.add_argument(
        "--ade",
        action='store_true',
        
        help="Flag for ade."
    )
    args = parser.parse_args()
    train_and_evaluate(args)

