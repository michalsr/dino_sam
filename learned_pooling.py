# %%
import os
import random
import pickle
import torch
from typing import Union
import numpy as np
import torch
import pickle
from utils import open_file, mean_iou
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb
from itertools import islice
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy
import logging, coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

# %%
class CustomDataset(Dataset):
    def __init__(self, sam_embed_dir, dino_feat_dir, train_region_labels_dir, max_pool_dir, sam_folder):
        super().__init__()
        self.sam_embed_files = [f for f in os.listdir(sam_embed_dir) if f.endswith('.pkl')]
        self.dino_feat_files = [f for f in os.listdir(dino_feat_dir) if f.endswith('.pkl')]
        self.label_files = [f for f in os.listdir(train_region_labels_dir) if f.endswith('.pkl')]
        self.sam_embed_dir = sam_embed_dir
        self.dino_feat_dir = dino_feat_dir
        self.train_region_labels_dir = train_region_labels_dir
        self.max_pool_dir = max_pool_dir
        self.sam_folder=sam_folder

    def __len__(self):
        return len(self.sam_embed_files)

    def __getitem__(self, idx):
        sam_embed_file = self.sam_embed_files[idx]
        dino_feat_file = self.dino_feat_files[idx]
        label_file = self.label_files[idx]

        # Load SAM region embeddings
        with open(os.path.join(self.sam_embed_dir, sam_embed_file), 'rb') as f:
            sam_embeds_all = pickle.load(f)

        # Load DINO whole image features
        with open(os.path.join(self.dino_feat_dir, dino_feat_file), 'rb') as f:
            dino_feats = pickle.load(f)

        # Load labels
        max_pool_masks = open_file(os.path.join(self.max_pool_dir, label_file))
        file_labels = open_file(os.path.join(self.train_region_labels_dir, label_file))

        # Filtering the SAM embeddings based on the presence of labels
        sam_embeds = []
        label_all = []
        for i, area in enumerate(max_pool_masks):
            area_label = file_labels[i]['labels']
            target_label = list(area_label.keys())[0]
            if area_label[target_label] == 1:
                label_all.append(target_label)
                sam_embeds.append(sam_embeds_all[i])

        # print(f"Index {idx} corresponds to files: {sam_embed_file}, {dino_feat_file}, {label_file}")
        # print(f"Data point {idx}: {len(sam_embeds)} SAM embeddings, {len(label_all)} labels")

        # Convert to numpy arrays first since torch applied to lists is slow
        sam_embeds = np.array(sam_embeds)
        label_all = np.array(label_all, dtype=np.int64)

        return torch.tensor(sam_embeds), torch.tensor(dino_feats), torch.tensor(label_all)

class AttentionSegmentation(nn.Module):
    def __init__(self, embed_dim, num_heads, num_classes):
        super(AttentionSegmentation, self).__init__()

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.1)

        # Final linear layer to output area-wise labels
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, sam_embeds, dino_feats):
        # Compute the average feature representation of DINO embeddings for the entire image
        #dino_avg = dino_feats.mean(dim=[-1, -2]).repeat(sam_embeds.shape[0], 1, 1)

        # sam_embeds are queries: shape [seq_len, batch_size, embed_dim]
        # dino_avg are keys and values: shape [seq_len, batch_size, embed_dim]
        attn_output, _ = self.attention(sam_embeds, dino_feats, dino_feats)

        # Predict area-wise labels
        output = self.fc(attn_output)

        return output

class AddDinoV2PosEmbeds(nn.Module):
    def __init__(self, dino_pos_embeds: torch.Tensor):
        super().__init__()
        self.pos_embeds = dino_pos_embeds.detach()

    def forward(self, dino_feats):
        '''
        Dino features are expected to be of shape (bsize, patches_h, patches_w, dim)
        where patches_h = padded_h // patch_size and patches_w = padded_w // patch_size.

        The CLS token should NOT be included.
        '''
        assert dino_feats.dim() == 4, 'Expected input of shape (bsize, patches_h, patches_w, dim))'
        interpolated_pos_embeds = self.interpolate_pos_encoding(dino_feats)

        return dino_feats + interpolated_pos_embeds

    def interpolate_pos_encoding(self, dino_feats: torch.Tensor):
        # Modified from https://github.com/facebookresearch/dinov2/blob/44abdbe27c0a5d4a826eff72b7c8ab0203d02328/dinov2/models/vision_transformer.py#L164
        previous_dtype = dino_feats.dtype
        npatch = dino_feats.shape[1] * dino_feats.shape[2]
        N = self.pos_embeds.shape[1] - 1 # Ignore CLS
        sqrt_N = np.sqrt(N)

        pos_embed = self.pos_embeds.float()
        patch_pos_embed = pos_embed[:, 1:] # Ignore CLS embedding
        dim = dino_feats.shape[-1]

        # Target height, width is height, width of the dino features
        h0 = dino_feats.shape[1]
        w0 = dino_feats.shape[2]

        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2), # (dim to second spot)
            scale_factor=(sy, sx),
            mode="bicubic",
        )

        assert int(h0) == patch_pos_embed.shape[-2]
        assert int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1) # (dim to end)

        return patch_pos_embed.to(previous_dtype)

def evaluate_model(adder, model, val_dataloader, num_classes):
    model.eval()  # Set the model to evaluation mode

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for sam_embeds, dino_feats, labels in tqdm(val_dataloader, desc='Evaluating model'):
            outputs = forward_pass(sam_embeds, dino_feats, model, adder)

            outputs = outputs.transpose(0, 1).view(-1, num_classes)
            _, predicted = torch.max(outputs.data, 1)
            labels = labels.view(-1).to(outputs.device)

            all_predictions.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return all_predictions, all_labels

def process_inputs(sam_embeds, dino_feats, adder: AddDinoV2PosEmbeds):
    dino_feats = dino_feats.to(adder.pos_embeds.device)
    sam_embeds = sam_embeds.to(adder.pos_embeds.device)

    dino_feats = dino_feats.squeeze(1)
    dino_feats = dino_feats.permute(0, 2, 3, 1)
    dino_feats = adder(dino_feats)
    dino_feats = dino_feats.permute(1, 2, 0, 3).reshape(-1, dino_feats.shape[0], embed_dim)
    sam_embeds = sam_embeds.transpose(0, 1)

    return sam_embeds, dino_feats

def forward_pass(sam_embeds, dino_feats, model, adder):
    sam_embeds, dino_feats = process_inputs(sam_embeds, dino_feats, adder)

    outputs = model(sam_embeds, dino_feats)
    outputs = outputs.transpose(0, 1) # (bsize, seq_len, num_classes)

    return outputs

def compute_mIOU(predictions, labels, num_classes):
    # Compute mIOU using predictions and ground truth annotations.
    num_classes = num_classes -1
    reduce_labels = True
    n_classes = num_classes - 1 if reduce_labels else num_classes
    miou = mean_iou(predictions, labels, num_labels=n_classes, ignore_index=255, reduce_labels=reduce_labels)

    logger.info(f'mIoU: {miou}')

    return miou

@torch.inference_mode()
def predict(
    dataloader,
    model,
    adder,
    return_outputs=False,
    criterion=None,
    metrics: Union[MetricCollection,Metric] = None
):
    model.eval()

    if metrics is not None:
        metrics.reset()

    all_outputs = []
    all_predictions = []
    all_labels = []
    loss = 0

    for sam_embeds, dino_feats, labels in tqdm(dataloader, desc='Predicting'):
        outputs = forward_pass(sam_embeds, dino_feats, model, adder)
        outputs = outputs.transpose(0, 1).view(-1, num_classes)
        _, predicted = torch.max(outputs.data, 1)

        labels = labels.view(-1).to(outputs.device)

        if return_outputs:
            all_outputs.append(outputs.cpu())

        if criterion is not None:
            loss += criterion(outputs, labels)

        if metrics is not None:
            metrics(predicted, labels)

        all_predictions.append(predicted.cpu())
        all_labels.append(labels.cpu())

    ret_dict = {
        'all_predictions': torch.cat(all_predictions),
        'all_labels': torch.cat(all_labels)
    }

    if return_outputs:
        ret_dict['all_outputs'] = torch.cat(all_outputs)

    if criterion is not None:
        ret_dict['loss'] = loss / len(dataloader)

    if metrics is not None:
        ret_dict['metrics'] = metrics.compute()

    return ret_dict

class FirstN:
    def __init__(self, iterable, n):
        self.iterable = iterable
        self.n = len(iterable) if n is None else min(n, len(iterable))

    def __len__(self):
        return self.n

    def __iter__(self):
        return islice(self.iterable, self.n)

# %%
if __name__ == '__main__':
    #  Hyperparameters
    device = 'cuda'
    accum_grad_steps = 4 # Number of images we average the gradients over before optimizer step
    limit_train_batches = None # None for all batches
    limit_val_batches = 500 # None for all batches
    lr = 5e-5
    n_epochs = 100
    seed = 42

    config = dict(
        device=device,
        accum_grad_steps=accum_grad_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        lr=lr,
        n_epochs=n_epochs,
        seed=seed
    )
    wandb.init(project='learned-pooling', config=config, reinit=True)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # %%
    embed_dim = 1024  # Adjusted feature dimension for sam_embeds and dino_feats
    num_heads = 8  # Number of heads in MultiheadAttention mechanism
    num_classes = 21  # Number of classes for area-wise labels

    # Model instantiation
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model = AttentionSegmentation(embed_dim, num_heads, num_classes).to(device)
    adder = AddDinoV2PosEmbeds(dino.pos_embed.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # %% Create dataloaders
    n_workers = 2

    sam_embed_dir="/shared/rsaas/dino_sam/sam_region_embeddings/pascal_voc/train_upscale"
    dino_feat_dir = "/shared/rsaas/dino_sam/features/dinov2/pascal_voc_layer_23/train"
    train_region_labels_dir = '/shared/rsaas/dino_sam/region_labels/pascal_voc/train'
    max_pool_dir = "/shared/rsaas/dino_sam/region_features/dinov2/pascal_voc_layer_23/train"
    sam_folder = "/shared/rsaas/dino_sam/sam_output/pascal_voc/train"

    dataset= CustomDataset(sam_embed_dir, dino_feat_dir, train_region_labels_dir, max_pool_dir, sam_folder)
    rng = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=n_workers, generator=rng)
    train_dataloader = FirstN(train_dataloader, limit_train_batches) # Potentially reduce number of batches

    # Validation
    sam_embed_val_dir = "/shared/rsaas/dino_sam/sam_region_embeddings/pascal_voc/val_upscale"
    dino_feat_val_dir = "/shared/rsaas/dino_sam/features/dinov2/pascal_voc_layer_23/val"
    train_region_labels_val_dir = '/shared/rsaas/dino_sam/region_labels/pascal_voc/val'
    max_pool_val_dir = "/shared/rsaas/dino_sam/region_features/dinov2/pascal_voc_layer_23/val"
    sam_val_folder = "/shared/rsaas/dino_sam/sam_output/pascal_voc/val"

    val_dataset = CustomDataset(sam_embed_val_dir, dino_feat_val_dir, train_region_labels_val_dir, max_pool_val_dir, sam_val_folder)
    rng = torch.Generator().manual_seed(seed)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=n_workers, generator=rng)
    val_dataloader = FirstN(val_dataloader, limit_val_batches) # Potentially reduce number of batches

    # %%
    # Training loop
    metrics = MetricCollection({
        'accuracy': MulticlassAccuracy(num_classes=num_classes, average='micro')
    }).to(device)

    for epoch in range(n_epochs):  # Example number of epochs
        logger.info(f'Starting training of epoch {epoch + 1}...')

        model.train()
        epoch_train_loss = 0

        prog_bar = tqdm(train_dataloader, desc='Training', total=len(train_dataloader))

        for i, (sam_embeds, dino_feats, labels) in enumerate(prog_bar, start=1):
            outputs = forward_pass(sam_embeds, dino_feats, model, adder)

            outputs = outputs.transpose(0, 1).view(-1, num_classes)
            labels = labels.view(-1).to(device)
            loss = criterion(outputs, labels)

            loss = loss / accum_grad_steps
            loss.backward()

            if i % accum_grad_steps == 0 or i == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
                epoch_train_loss += loss.item()
                wandb.log({'train_loss': loss.item()})

        epoch_train_loss = epoch_train_loss / (len(train_dataloader) / accum_grad_steps)
        wandb.log({'epoch_train_loss': epoch_train_loss, 'epoch': epoch + 1})

        # Compute validation loss
        logger.info('Starting validation set evaluation...')

        results = predict(val_dataloader, model, adder, criterion=criterion, metrics=metrics)
        val_loss = results['loss']
        val_accuracy = results['metrics']['accuracy']
        wandb.log({'val_loss': val_loss, 'val_accuracy': val_accuracy, 'epoch': epoch + 1})

        # NOTE This should not be used because here we're classifying regions, NOT computing segmentation maps
        # predictions, labels = evaluate_model(adder, model, val_dataloader, num_classes)
        # miou = compute_mIOU(predictions, labels, num_classes)

        # wandb.log({
        #     'mIoU': miou['mean_iou'],
        #     'mean_accuracy': miou['mean_accuracy'],
        #     'overall_accuracy': miou['overall_accuracy'],
        #     'epoch': epoch + 1
        # })

        logger.info(f"Epoch {epoch + 1}, Train Loss: {epoch_train_loss:.3f}, Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}")
# %%
