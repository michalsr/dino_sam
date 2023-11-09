import socket
import glob
from torch.utils.data import Dataset
import gzip
import pickle
import os
from tqdm import tqdm
import itertools
import argparse
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from timm.models.layers import trunc_normal_
import torch.optim as optim
from utils import *
import time
import os.path as osp



def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class DefaultResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, key_padding_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.key_padding_mask = key_padding_mask


    def attention(self, x: torch.Tensor):
        self.key_padding_mask = self.key_padding_mask.to(dtype=x.dtype, device=x.device) if self.key_padding_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=self.key_padding_mask)[0]


    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class custom_transformer(nn.Module):
    def __init__(self, region_num: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()

        scale = width ** -0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(region_num + 1, width))
        # self.ln_pre = LayerNorm(width)
        self.layers = layers

        self.transformer = nn.Sequential(*[
            DefaultResidualAttentionBlock(width, heads) for i in range(layers)])

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.softmax = nn.Softmax(dim=-1)


    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, DefaultResidualAttentionBlock):
                m.position_embeddings.data = nn.init.trunc_normal_(
                    m.position_embeddings.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(m.position_embeddings.dtype)

                m.cls_token.data = nn.init.trunc_normal_(
                    m.cls_token.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(m.cls_token.dtype)
        self.apply(_init_weights)


    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        # the shape of x is [*, region_num, width]
        x = torch.cat([self.cls_token.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, region_num + 1, width]
        # x = x + self.positional_embedding.to(x.dtype)
        # x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.layers):
            self.transformer[i].key_padding_mask = key_padding_mask

        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 1:, :])

        if self.proj is not None:
            x = x @ self.proj
        return x
        return self.softmax(x)


class MLP(nn.Module):
    def __init__(self, feature_length, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_length, output_dim),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x, key_padding_masks = None):
        return self.model(x)


class load_dataset(Dataset):
    def __init__(self, root, region_num, model_type, split, scene_level='image', feature='dino_feat'):
        super().__init__()
        self.root = Path(root) / split
        self.region_num = region_num
        self.model_type = model_type
        self.split = split
        self.scene_level = scene_level

        self.feature_file_paths = self.get_feature_paths(feature, 'json')
        # self.feature_file_paths = self.get_dense_clip_feature_json()
        self.pos_encoding_file_paths = self.get_feature_paths('pos_encoding', 'pth')
        self.label_file_paths = self.get_label_paths()

        scene_image_num = [len(sorted((i / 'color').iterdir())) for i in (sorted(self.root.iterdir()))]
        region_features, pos_encodings, image_pos_encodings, labels, image_region_num, image_names = self.load_data()


        self.samples = []
        if self.scene_level == 'image':
            if self.model_type == 'MLP':
                for index in range(len(region_features)):
                    sample = {}
                    sample['region_feature'] = region_features[index]
                    sample['pos_encoding'] = pos_encodings[index]
                    sample['image_pos_encoding'] = image_pos_encodings[index]
                    sample['label'] = labels[index]
                    # sample['image_name'] = image_names[index]
                    self.samples.append(sample)
            else:
                image_region_num_accumulate = np.add.accumulate(np.array(image_region_num)).tolist()
                image_region_num_accumulate = [0] + image_region_num_accumulate

                for i, region_num in enumerate(image_region_num_accumulate):
                    sample = {}
                    if i == len(image_region_num_accumulate) - 1:
                        break

                    region_torch = region_features[region_num: image_region_num_accumulate[i+1]]
                    pos_encoding_torch = pos_encodings[region_num: image_region_num_accumulate[i+1]]
                    image_pos_encoding_torch = image_pos_encodings[region_num: image_region_num_accumulate[i+1]]
                    label_torch = labels[region_num: image_region_num_accumulate[i+1]]

                    sample['region_feature'] = region_torch
                    sample['pos_encoding'] = pos_encoding_torch
                    sample['image_pos_encoding'] = image_pos_encoding_torch
                    sample['label'] = label_torch
                    sample['image_name'] = image_names[i]
                    self.samples.append(sample)

        elif self.scene_level == 'scene':
            scene_image_num_accumulate = np.add.accumulate(np.array(scene_image_num)).tolist()
            scene_image_num_accumulate = [0] + scene_image_num_accumulate
            scene_region_num = []
            for i, scene_image_num in enumerate(scene_image_num_accumulate):
                if i == len(scene_image_num_accumulate) - 1:
                    break
                scene_region_num.append(sum(image_region_num[scene_image_num: scene_image_num_accumulate[i+1]]))
            scene_region_num_accumulate = np.add.accumulate(np.array(scene_region_num)).tolist()
            scene_region_num_accumulate = [0] + scene_region_num_accumulate

            for i, scene_image_num in enumerate(scene_region_num_accumulate):
                sample = {}
                if i == len(scene_region_num_accumulate) - 1:
                    break
                region_torch = region_features[scene_image_num: scene_region_num_accumulate[i+1]]
                pos_encoding_torch = pos_encodings[scene_image_num: scene_region_num_accumulate[i+1]]
                image_pos_encoding_torch = image_pos_encodings[scene_image_num: scene_region_num_accumulate[i+1]]
                label_torch = labels[scene_image_num: scene_region_num_accumulate[i+1]]

                sample['region_feature'] = region_torch
                sample['pos_encoding'] = pos_encoding_torch
                sample['image_pos_encoding'] = image_pos_encoding_torch
                sample['label'] = label_torch
                sample['image_name'] = image_names[i]
                self.samples.append(sample)
        else:
            raise NotImplementedError


    def load_data(self,):
        region_features = []
        pos_encodings = []
        image_pos_encodings = []
        labels = []
        image_region_num = []
        image_names = []

        for index in tqdm(range(len(self.label_file_paths)), leave='load samples'):
            feature_file = self.feature_file_paths[index]
            label_file = self.label_file_paths[index]
            pos_encoding_file = self.pos_encoding_file_paths[index]
            image_names.append(str(feature_file))

            label = open_file(label_file)
            feature = open_file(feature_file)
            pos_encoding = torch.load(pos_encoding_file)

            region_count = 0
            for j, region_feature in enumerate(feature):
                region_label = label[j]['labels']
                region_pos_encoding = torch.cat([pos_encoding[j]['x_pos_enc'], 
                                                    pos_encoding[j]['y_pos_enc'], 
                                                    pos_encoding[j]['z_pos_enc']]) 
                region_image_pos_encoding = torch.cat([pos_encoding[j]['u_pos_enc'], 
                                                    pos_encoding[j]['v_pos_enc'],]) 
                target_label = list(region_label.keys())[0]

                if not region_label[target_label] == 1: # Regions without valid label
                    # if self.split == 'train':
                    #     continue
                    target_label = -1 # Mark as invalid label, would be ignored during the loss

                # Some regions don't have valid 3d points, so have nan 3d pos encoding
                if torch.isnan(region_pos_encoding).any(): 
                    continue
                    
                region_features.append(torch.tensor(region_feature['feature']))
                pos_encodings.append(region_pos_encoding)
                image_pos_encodings.append(region_image_pos_encoding)
                labels.append(target_label)
                region_count += 1
            if region_count > 0:
                image_region_num.append(region_count)

        region_features = torch.stack(region_features)
        pos_encodings = torch.stack(pos_encodings)
        image_pos_encodings = torch.stack(image_pos_encodings)
        labels = torch.tensor(labels)
        return region_features, pos_encodings, image_pos_encodings, labels, image_region_num, image_names     

    def get_feature_paths(self, feature, file_type):
        feature_list = []
        for scene_dir in sorted(self.root.iterdir()):
            feature_dir = scene_dir / feature / file_type
            for feature_file in sorted(feature_dir.iterdir()):
                feature_list.append(feature_file)
        return feature_list

    def get_label_paths(self,):
        label_list = []
        for scene_dir in sorted(self.root.iterdir()):
            label_dir = scene_dir / 'region-labels'
            for label_file in sorted(label_dir.iterdir()):
                label_list.append(label_file)
        return label_list        
    

    def __getitem__(self, index):
        sample = self.samples[index]
        region_feature = sample['region_feature']
        pos_encoding = sample['pos_encoding']
        image_pos_encoding = sample['image_pos_encoding']
        label = sample['label']
        if self.model_type == 'transformer':
            if region_feature.shape[0] > self.region_num:
                region_feature = region_feature[:self.region_num, :]
                pos_encoding = pos_encoding[:self.region_num, :]
                image_pos_encoding = image_pos_encoding[:self.region_num, :]
                label = label[:self.region_num]
                key_padding_mask = torch.zeros(self.region_num + 1)
            else:
                attn_num = region_feature.shape[0]

                pad_feature = torch.zeros((self.region_num - attn_num, region_feature.shape[1]))
                region_feature = torch.cat((region_feature, pad_feature), dim=0)

                pad_pos_encoding = torch.zeros((self.region_num - attn_num, pos_encoding.shape[1]))
                pos_encoding = torch.cat((pos_encoding, pad_pos_encoding), dim=0)

                pad_image_pos_encoding = torch.zeros((self.region_num - attn_num, image_pos_encoding.shape[1]))
                image_pos_encoding = torch.cat((image_pos_encoding, pad_image_pos_encoding), dim=0)

                label = torch.cat((label, -torch.ones(self.region_num - attn_num).int()), dim=0)
                key_padding_mask = torch.tensor(np.concatenate([[0] * (attn_num + 1), [1] * (self.region_num - attn_num)]))
        else:
            key_padding_mask = 0
        if (label!= -1).sum() == 0:
            pass
        sample['key_padding_mask'] = key_padding_mask
        sample['region_feature'] = region_feature
        sample['pos_encoding'] = pos_encoding
        sample['image_pos_encoding'] = image_pos_encoding
        sample['label'] = label

        return sample


    def __len__(self):
        return len(self.samples)



def collate_fn(batch):
    [region_features, labels] = zip(*batch)
    return region_features, labels



def init_loader(dataset, batch_size=8, shuffle=False):
    image_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        # collate_fn=collate_fn,
    )
    return image_loader


def load_model(args):
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    feature_length = args.feature_length
    if args.use_scene_pos_encoding:
        feature_length += 3*8*2
    if args.use_image_pos_encoding:
        feature_length += 2*8*2

    if args.model_type == 'MLP':
        model = MLP(feature_length, args.output_dim)
    elif args.model_type == 'transformer':
        model = custom_transformer(args.region_num, feature_length, args.layers, args.heads, args.output_dim)
    else:
        raise NotImplementedError

    if args.load_model_dir is not None:
        model.load_state_dict(torch.load(args.load_model_dir))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-1).cuda()

    model.to(device=device)
    model.train()
    return model, device, optimizer, criterion


def main_eval(args, model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    print('evaluating...')
    for i, sample in enumerate(tqdm(test_loader)):
        region_features, pos_encoding, image_pos_encoding, labels = sample['region_feature'], sample['pos_encoding'], sample['image_pos_encoding'], sample['label']
        key_padding_masks = sample['key_padding_mask']

        if args.use_scene_pos_encoding:
            region_features = torch.cat((region_features, pos_encoding), dim=-1)
        if args.use_image_pos_encoding:
            region_features = torch.cat((region_features, image_pos_encoding), dim=-1)   

        output = model(region_features.float().to(device=device), -key_padding_masks * 10 ** 8)
        # _, pre = torch.max(output.data, 1)
        # pre = pre.squeeze()
        # correct += (pre.cpu() == labels).sum().item()
        _, pre = torch.max(output.data, -1)
        pre = pre.squeeze()
        labels = labels.squeeze()
        label_valid_mask = (labels != -1).squeeze()
        correct += (pre.cpu()[label_valid_mask] == labels[label_valid_mask]).sum().item()  
        total += label_valid_mask.sum().item()
    eval_acc = correct / total
    print('acc on val dataset: %.2f' % (eval_acc))
    return eval_acc


def main_train(args, model, device, optimizer, criterion, log_path, log_eval_path):

    root = args.base_dir
    train_dataset = load_dataset(root, args.region_num, args.model_type, split = 'train', scene_level = args.scene_level)
    test_dataset = load_dataset(root, args.region_num, args.model_type, split = 'val', scene_level = args.scene_level)
    train_loader = init_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
    num_batches = len(train_loader)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_epochs * num_batches, args.train_epochs * num_batches)



    for epoch in range(args.train_epochs):
        model.train()
        train_loader = init_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
        training_correct_count = 0
        training_total_count = 0
        for i, sample in enumerate(tqdm(train_loader)):
            region_features, pos_encoding, image_pos_encoding, labels = sample['region_feature'], sample['pos_encoding'], sample['image_pos_encoding'], sample['label']
            if (labels != -1).sum() == 0:
                continue
            key_padding_masks = sample['key_padding_mask']
            # image_name = sample['image_name']

            if args.use_scene_pos_encoding:
                region_features = torch.cat((region_features, pos_encoding), dim=-1)
            if args.use_image_pos_encoding:
                region_features = torch.cat((region_features, image_pos_encoding), dim=-1)
            
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            output = model(region_features.float().to(device=device), -key_padding_masks * 10**8)
            _, pre = torch.max(output.data, -1)
            if not args.scene_level == 'scene':
                pre = pre.squeeze()
            label_valid_mask = (labels != -1)
            correct = (pre.cpu()[label_valid_mask] == labels[label_valid_mask]).sum().item()
            acc = correct / label_valid_mask.sum().item()
            training_correct_count += correct
            training_total_count += label_valid_mask.sum().item()

            if output.dim() == 3: 
                # output: [batch_size, region_num, output_dim], label: [batch_size, region_num]
                loss = criterion(output.permute(1, 2, 0), labels.permute(1, 0).long().to(device=device))
            else:
                # output: [batch_size, output_dim], label: [batch_size]
                loss = criterion(output, labels.long().to(device=device))
            loss.backward()
            optimizer.step()

            print('\n epoch: %d, lr: %.5f, loss: %.2f, acc: %.2f'%(epoch, optimizer.param_groups[0]['lr'], loss.item(), acc))
            with open(log_path, 'a') as f:
                f.write(f'\n epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"]}, loss: {loss.item()}, acc: {acc}')

        train_acc_total = training_correct_count / training_total_count
        with open(log_eval_path, 'a') as f:
            f.write(f'\n epoch: {epoch}, training total acc: {train_acc_total}')        
        if epoch % args.eval_and_save_epochs == 0:
            test_dataloader = init_loader(test_dataset, batch_size=args.batch_size, shuffle=False)
            eval_acc = main_eval(args, model, device, test_dataloader)
            with open(log_eval_path, 'a') as f:
                f.write(f'\n epoch: {epoch}, evaluation acc: {eval_acc}')

            # model_path = os.path.join(args.save_model_root, f'checkpoint_{epoch + 1}_eval_acc_{round(eval_acc, 2)}.pt')
            # print('Saving model to', model_path)
            # torch.save(model.state_dict(), model_path)
            # optim_path = os.path.join(args.save_model_root, f'optim_{epoch + 1}.pt')
            # torch.save(optimizer.state_dict(), optim_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default='MLP', help="transformer/MLP")
    parser.add_argument("--output_dim", type=int, default=41)
    parser.add_argument("--feature_length", type=int, default=1024)
    parser.add_argument("--region_num", type=int, default=50)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--warmup_epochs', default=2, type=int)

    parser.add_argument('--train_epochs', default=10, type=int)
    parser.add_argument('--eval_and_save_epochs', default=1, type=int)
    parser.add_argument('--load_model_dir', default=None, type=str)
    parser.add_argument('--save_model_root', default='checkpoints', type=str)


    parser.add_argument('--base_dir', default='data', type=str)
    parser.add_argument('--scene_level', default='image', type=str, choices=['image', 'scene'])
    parser.add_argument('--use_scene_pos_encoding', default=False, action='store_true')
    parser.add_argument('--use_image_pos_encoding', default=False, action='store_true')

    

    args = parser.parse_args()


    

    for k, v in vars(args).items():
        print(f'{k}: {v}')
    if args.use_scene_pos_encoding:
        use_scene_pos_encoding = '_use_scene_pos_encoding'
    else:
        use_scene_pos_encoding = ''
    if args.use_image_pos_encoding:
        use_image_pos_encoding = '_use_image_pos_encoding'
    else:
        use_image_pos_encoding = ''
    args.save_model_root = os.path.join(args.save_model_root, args.model_type,
                                        f'epoch_{args.train_epochs}_lr_{args.lr}_batch_{args.batch_size}_'
                                            f'region_num_{args.region_num}_heads_{args.heads}_layers_'
                                                f'{args.layers}_output_dim_{args.output_dim}_'
                                                    f'{args.scene_level}{use_scene_pos_encoding}{use_image_pos_encoding}')
    os.makedirs(args.save_model_root, exist_ok=True)

    args_path = os.path.join(args.save_model_root, 'args.txt')
    with open(args_path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    model, device, optimizer, criterion = load_model(args)
    log_path = os.path.join(args.save_model_root, 'log.txt')
    log_eval_path = os.path.join(args.save_model_root, 'log_eval.txt')
    main_train(args, model, device, optimizer, criterion, log_path, log_eval_path)
    # main_eval(args, model, device)