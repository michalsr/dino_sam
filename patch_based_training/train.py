import argparse
import logging
import os
from typing import List, Tuple, Literal
import time
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import coloredlogs
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassAccuracy
from tqdm import tqdm

import utils
from patch_based_training.linear_head import LinearHead
from utils import mean_iou

'''
Use transformer to classify regions
'''

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=logger)

def collate_fn(batch: List[Tuple]):
    features_l, label_ims_l = tuple(zip(*batch))

    return features_l, label_ims_l

class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_feat_dir: str,
        seg_label_dir: str = None,
        random_seed = 42,
        limit_files_to_frac = 1.0,
        model_name: Literal['dinov2', 'dinov1', 'clip', 'denseclip']= 'dinov2'
    ):
        super().__init__()

        self.rng = np.random.default_rng(random_seed)
        self.image_feat_dir = image_feat_dir
        self.seg_label_dir = seg_label_dir
        self.model_name = model_name

        self.image_feat_paths = sorted([
            os.path.join(image_feat_dir, file_name)
            for file_name in os.listdir(image_feat_dir)
        ])

        self.rng.shuffle(self.image_feat_paths)

        if limit_files_to_frac < 1:
            self.image_feat_paths = self.image_feat_paths[:int(len(self.image_feat_paths) * limit_files_to_frac)]

    def __len__(self):
        return len(self.image_feat_paths)

    def __getitem__(self, idx):

        image_feat_path = self.image_feat_paths[idx]
        image_feat = utils.open_file(image_feat_path)
        if isinstance(image_feat, np.ndarray):
            image_feat = torch.from_numpy(image_feat)

        # TODO add reshaping cases for all models' features
        if self.model_name == 'dinov2':
            pass # Already in shape (1, c, nph, npw)
        elif self.model_name == 'clip':
            n,hw,c = image_feat.size()
            image_feat = image_feat.permute(0, 2, 1).view(n,c,7,7)
            #image_feat = rearrange(image_feat, 'n b d -> n h e d',h=7,e=7) # ViT-L/14
        elif self.model_name == 'dinov1':
            pass # Already in shape (1, c, nph, npw)
        elif self.model_name == 'denseclip':
            pass # Already in shape (1, c, nph, npw)
        else:
            raise NotImplementedError(f'Not implemented for model {self.model_name}')

        if self.seg_label_dir is not None:
            seg_label_path = os.path.join(self.seg_label_dir, os.path.basename(image_feat_path).replace('.pkl', '.png'))
            seg_label = torch.from_numpy(np.array(Image.open(seg_label_path)))
        else:
            seg_label = None

        return image_feat, seg_label

def resize_outputs_to_image_size(outputs: torch.Tensor, images: List[torch.Tensor]):
    '''
    Args:
        outputs (torch.Tensor): (n, h, w, n_classes)
        images (List[torch.Tensor]): List of images of shape (h, w, c)

    Returns:
        List[torch.Tensor]: List of resized outputs of shape (h_i, w_i, n_classes)
    '''
    outputs = rearrange(outputs, 'n h w c -> n c h w')
    image_l = [
        F.interpolate(
            output[None,...], # (1, c, h, w); expects batch and channel dimensions to lead
            size=image.shape[-2:],
            mode="bilinear"
        ).squeeze()
        if output.shape[-2:] != image.shape[-2:] else output
        for output, image in zip(outputs, images)
    ]
    image_l = [
        rearrange(image, 'c h w -> h w c')
        for image in image_l
    ]

    return image_l

def get_model_outputs(feats_l, labels_l, model):
    feats_l = [f.to(args.device) for f in feats_l]
    labels_l = [l.to(args.device) for l in labels_l]
    outputs = model(feats_l) # (n, h, w, n_classes)
    image_preds_l = resize_outputs_to_image_size(outputs, labels_l)

    return image_preds_l, labels_l

def compute_loss(image_preds_l, labels_l, criterion):
    preds = torch.cat([
        img_preds.reshape(-1, img_preds.shape[-1])
        for img_preds in image_preds_l
    ], dim=0) # (n, n_classes)

    labels = torch.cat([
        labels.reshape(-1)
        for labels in labels_l
    ], dim=0)

    loss = criterion(preds, labels)

    return loss, preds, labels

def get_dataloader(ds: SegmentationDataset, is_train: bool, args: argparse.Namespace):
    return DataLoader(
        ds,
        batch_size=args.batch_size if is_train else 1,
        shuffle=is_train,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )

def train_model(args):
    # Set up dataset, model, optimizmer, scheduler, loss
    dataloader = get_dataloader(args.train_ds, is_train=True, args=args)

    model = LinearHead(args.num_classes + 1).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)

    if args.ade:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        mca = MulticlassAccuracy(num_classes=args.num_classes+1, average='micro', top_k=1, ignore_index=0).to(args.device)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        mca = MulticlassAccuracy(num_classes=args.num_classes+1, average='micro', top_k=1, ignore_index=255).to(args.device)

    epochs = args.epochs

    logger.info(f'Train dataloader length with batch size {args.batch_size}: {len(dataloader)}')

    # Start training
    for i,epoch in enumerate(range(epochs)):
        model.train()
        mca.reset()

        for feats_l, labels_l in tqdm(dataloader, desc=f'Train epoch {epoch + 1}'):
            image_preds_l, labels_l = get_model_outputs(feats_l, labels_l, model)

            loss, preds, labels = compute_loss(image_preds_l, labels_l, criterion)
            
            loss = loss/args.accumulate_grad_batches
            loss.backward()
            
            if i%args.accumulate_grad_batches == 0 or i==len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            mca(preds.detach(), labels)

        logger.info(f"Epoch {epoch + 1} train loss: {loss.item()}")

        torch.save(model.cpu().state_dict(), os.path.join(args.save_dir, 'model.pt'))

        logger.info('Computing validation metrics')
        val_loss, val_acc = eval_acc(args, model)
        metrics = {'val_loss': val_loss,'val_acc': val_acc, 'train_acc': mca.compute().item(), 'train_loss': loss.item()}
        utils.save_file(os.path.join(args.results_dir, f'metrics_epoch_{epoch}.json'), metrics, json_numpy=True)

        scheduler.step()

        if args.log_to_wandb:
            metrics.update({
                'epoch': epoch,
                'lr': scheduler.get_last_lr()[0]
            })
            wandb.log(metrics)

        if (epoch + 1) % args.iou_every==0:
            all_pixel_predictions, file_names = predict_model(args)
            compute_iou(args,all_pixel_predictions, file_names, epoch)

# Can't use inference mode inside of training loop
@torch.no_grad()
def eval_acc(args, model: torch.nn.Module):
    model.to(args.device).eval()
    dataloader = get_dataloader(args.val_ds, is_train=False, args=args)

    if args.ade:
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        mca = MulticlassAccuracy(num_classes=args.num_classes + 1, average='micro', top_k=1, ignore_index=0).to(args.device)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        mca = MulticlassAccuracy(num_classes=args.num_classes+1, average='micro', top_k=1, ignore_index=255).to(args.device)

    total_loss = 0
    for feats_l, labels_l in tqdm(dataloader):
        assert len(feats_l) == 1
        image_preds_l, labels_l = get_model_outputs(feats_l, labels_l, model)
        loss, preds, labels = compute_loss(image_preds_l, labels_l, criterion)

        # Reshape outputs and labels for loss calculation
        mca(preds.cpu() labels.cpu())
        total_loss += loss.item()

    val_loss = total_loss / len(args.val_ds) # Average loss per image
    logger.info(f'Val loss: {val_loss}')

    val_acc = mca.compute().item()
    logger.info(f'Val acc: {val_acc}')

    return val_loss, val_acc

@torch.inference_mode()
def predict_model(args):
    logger.info('Generating model predictions')
    model = LinearHead(args.num_classes + 1)

    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'model.pt')))
    model.to(args.device).eval()

    dataloader = get_dataloader(args.val_ds, is_train=False, args=args)
    predictions = []
    for feats_l, labels_l in tqdm(dataloader):
        assert len(feats_l) == 1
        image_preds_l, _ = get_model_outputs(feats_l, labels_l, model) # List of (h, w, n_classes)
        image_preds = image_preds_l[0].argmax(dim=-1).cpu() # (h, w)
        predictions.append(image_preds)

    img_ext = '.jpg' if args.ade else '.png'
    file_names = [os.path.basename(p).replace('.pkl', img_ext) for p in args.val_ds.image_feat_paths]

    return predictions, file_names

def compute_iou(args, predictions, file_names, epoch):
    logger.info('Computing IoU')
    actual_labels = []
    for file in tqdm(file_names):
        if args.ade:
            file = file.replace('.jpg','.png')
        actual = np.array(Image.open(os.path.join(args.val_seg_label_dir, file)))
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
    logger.info(miou)
    utils.save_file(os.path.join(args.results_dir, f'mean_iou_epoch_{epoch}.json'), miou, json_numpy=True)

def train_and_evaluate(args):
    if args.num_classes != 150 and args.num_classes!= 20:
        raise ValueError('ADE should have 150 and Pascal VOC should have 20. The background class is taken care of in the code')
    if args.num_classes == 150:
        if args.ade ==False:
            raise ValueError('If using ADE then ade argument should be set to True')
    if args.ade==True:
        logger.info('Training and evaluating on ADE. Make sure to use the correct region label directory (ADE20K_no_zero)!')
    if not args.eval_only:
        train_model(args)

    all_pixel_predictions, file_names = predict_model(args)

    # Save pixel predictions as PNGs for use on evaluation server
    if args.output_predictions:
        logger.info('Saving predictions to PNGs')
        prediction_dir = os.path.join(args.results_dir, 'predictions')
        os.makedirs(prediction_dir, exist_ok=True)

        for file_name, prediction in tqdm(zip(file_names, all_pixel_predictions)):
            prediction = Image.fromarray(prediction.astype(np.uint8))
            prediction.save(os.path.join(prediction_dir, file_name.replace('.pkl', '.png')))

    if not args.no_evaluation: # No need to output predictions if evaluating here
        compute_iou(args, all_pixel_predictions, file_names, args.epochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backbone_model',
        type=str,
        choices=['dinov2', 'dinov1', 'clip', 'denseclip'],
        help='Backbone model which produced the features, in order to reshape them to the correct dimensions'
    )

    parser.add_argument(
        '--train_img_feature_dir',
        type=str,
        default=None,
        help='Location of the training images\' patch features'
    )

    parser.add_argument(
        '--train_seg_label_dir',
        type=str,
        default=None,
        help='Location of the training images\' segmentation labels'
    )

    parser.add_argument(
        '--val_img_feature_dir',
        type=str,
        help='Location of the validation images\' patch features'
    )

    parser.add_argument(
        '--val_seg_label_dir',
        type=str,
        default=None,
        help='Location of the validation images\' segmentation labels'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of iterations to run log regression'
    )

    parser.add_argument(
        "--ignore_zero",
        action="store_true",
        help="Include 0 class"
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
        "--lr",
        type=float,
        default=.0001,
        help="learning rate"
    )

    parser.add_argument(
        "--multi_region_pixels",
        type=str,
        default="avg_after_softmax",
        help="What to do for pixels in multiple regions. Default is average over probabilities after softmax"
    )

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
        '--output_predictions',
        action='store_true',
        help='Output predictions as PNGs'
    )

    parser.add_argument(
        '--ade',
        action='store_true',
        help='Whether the datset we\'re running on is ADE20K. Adjusts labeling and loss computation'
    )

    parser.add_argument(
        '--override_ade_detection',
        action='store_true',
        help='Whether to train/eval anyways in spite of the ADE20K dataset detection.'
    )

    parser.add_argument(
        '--no_evaluation',
        action='store_true',
        help='Whether to skip evaluation (e.g. for Pascal VOC test which hasn\'t released labels.'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=2,
        help='Number of dataloader workers'
    )

    parser.add_argument(
        '--log_to_wandb',
        action='store_true',
        help='Whether to log results to wandb'
    )
    parser.add_argument(
        '--limit_files_to_frac',
        type=float,
        default=1.0,
        help='Limit the number of files to this fraction of the total'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to perform gradient accumulation over."
    )
    args = parser.parse_args()

    # Set run name as YYYY-MM-DD_HH-MM-SS
    run_name = time.strftime("%Y-%m-%d_%H-%M-%S")
    # args = parser.parse_args([
    #     # '--train_img_feature_dir', '/shared/rsaas/dino_sam/features/dinov2/ADE20K/train',

    #     # Pascal parameters
    #     '--train_img_feature_dir', '/shared/rsaas/dino_sam/features/denseclip/pascal_voc/train',
    #     '--train_seg_label_dir', '/shared/rsaas/dino_sam/data/VOCdevkit/VOC2012/segmentation_annotation/train',

    #     '--val_img_feature_dir', '/shared/rsaas/dino_sam/features/denseclip/pascal_voc/val',
    #     '--val_seg_label_dir', '/shared/rsaas/dino_sam/data/VOCdevkit/VOC2012/segmentation_annotation/val',

    #     '--num_classes', '20',

    #     # General parameters
    #     '--backbone_model', 'denseclip',
    #     '--epochs', '20',

    #     '--save_dir', f'/home/blume5/shared/dinov1_linear/pascal_voc/runs/{run_name}/checkpoints',
    #     '--results_dir', f'/home/blume5/shared/dinov1_linear/pascal_voc/runs/{run_name}/results',

    #     '--lr', '1e-3',
    #     '--batch_size', '8',
    #     '--iou_every', '10000',
    #     '--log_to_wandb',
    # ])

    # args = parser.parse_args([
    #     # '--train_img_feature_dir', '/shared/rsaas/dino_sam/features/dinov2/ADE20K/train',

    #     # Pascal parameters
    #     '--train_img_feature_dir', '/shared/rsaas/dino_sam/features/dinov1/pascal_voc_layer_11/train',
    #     '--train_seg_label_dir', '/shared/rsaas/dino_sam/data/VOCdevkit/VOC2012/segmentation_annotation/train',

    #     '--val_img_feature_dir', '/shared/rsaas/dino_sam/features/dinov1/pascal_voc_layer_11/val',
    #     '--val_seg_label_dir', '/shared/rsaas/dino_sam/data/VOCdevkit/VOC2012/segmentation_annotation/val',

    #     '--num_classes', '20',

    #     # General parameters
    #     '--backbone_model', 'dinov1',
    #     '--epochs', '20',

    #     '--save_dir', f'/home/blume5/shared/dinov1_linear/pascal_voc/runs/{run_name}/checkpoints',
    #     '--results_dir', f'/home/blume5/shared/dinov1_linear/pascal_voc/runs/{run_name}/results',

    #     '--lr', '1e-3',
    #     '--batch_size', '8',
    #     '--iou_every', '10000',
    #     '--log_to_wandb',
    # ])

    # Try to detect whether the dataset is ADE20K, and if so, force the user to set the flag
    dirs = [
        s.lower() for s in [
            args.train_seg_label_dir,
            args.val_seg_label_dir,
            args.train_img_feature_dir,
            args.val_img_feature_dir
        ]
    ]

    if 'ade20k' in dirs and not args.override_ade_detection:
        raise ValueError('Detected ADE20K dataset. Please set the --ade flag.')

    if not args.eval_only and args.train_img_feature_dir is None or args.train_seg_label_dir is None:
        raise ValueError('Must provide training image features and segmentation labels')

    # Save arguments to save_dir
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    if args.log_to_wandb:
        import wandb
        wandb.init(project='dino_sam_patch_pred', config=args)

    args.val_ds = SegmentationDataset(args.val_img_feature_dir, args.val_seg_label_dir,model_name=args.backbone_model)
    if not args.eval_only:
        args.train_ds = SegmentationDataset(args.train_img_feature_dir, args.train_seg_label_dir, limit_files_to_frac=args.limit_files_to_frac,model_name=args.backbone_model)

    train_and_evaluate(args)
