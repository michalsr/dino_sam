'''
Generates embeddings for SAM regions as the average of the positional embeddings of the patches
containing the downsampled SAM regions.
'''
# %%
import os
import pickle
import json
from typing import List
from pycocotools import mask as mask_utils
from einops import rearrange, reduce
import torch
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import itertools
import math
import argparse
from tqdm import tqdm
import logging, coloredlogs

logger = logging.getLogger(__name__)
coloredlogs.install(level=logging.INFO, logger=logger)

class CenterPadding(torch.nn.Module):
    '''
    Padding method for DinoV2. Copied from:
    https://github.com/facebookresearch/dinov2/blob/44abdbe27c0a5d4a826eff72b7c8ab0203d02328/dinov2/hub/utils.py
    '''
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output

class RegionEmbeddingGenerator:
    def __init__(
        self,
        dino_feats: torch.FloatTensor,
        sam_masks: torch.IntTensor,
        dinov2: torch.nn.Module,
        padder: CenterPadding
    ):
        '''
        Args:
            dino_feats (torch.Tensor): (d, npatches_h, npatches_w)
            sam_masks (torch.Tensor): (nmasks, h, w),
            dinov2 (torch.nn.Module): Dinov2 model used to generate the features.
            padder (torch.nn.Module): Padding module used for the corresponding Dinov2 model.
        '''
        # out-channels of the patch convolution is the feature dimension
        if dino_feats.shape[0] != dinov2.patch_embed.proj.out_channels:
            raise ValueError('Feature dimension of dino_feats must match out_channels of Dinov2 patch_embed')

        self.dino_feats = dino_feats
        self.sam_masks = sam_masks
        self.dinov2 = dinov2
        self.padder = padder

        # CenterPadding expects (b, c, h, w), so add a channel dimension then squeeze it out
        self.padded_sam_masks = padder(sam_masks.unsqueeze(1)).squeeze() # (nmasks, padded_h, padded_w)

    def downsample_positional_embeddings(self) -> torch.FloatTensor:
        '''
        Downsamples DinoV2's positional embeddings to match the dimension of the patch features.

        Based on interpolate_pos_encoding and prepare_tokens_with_masks of DinoVisionTransformer class

        Returns:
            torch.FloatTensor: (npatches_h, npatches_w, dim)
        '''
        # Note that codebase names tensor dimensions as (B, nc, w, h), but based on the output Dino feature dimensions and the
        # width and height of the original image, we believe w h > h w is a naming convention, and it is actually trained for h w.
        dino_feats = self.dino_feats
        dinov2 = self.dinov2
        sam_masks = self.padded_sam_masks

        npatch = dino_feats.shape[1] * dino_feats.shape[2] # Total number of patches
        h, w = sam_masks.shape[1:]

        pos_embed = dinov2.pos_embed.data # Extract tensor from parameter
        N = pos_embed.shape[1] - 1 # Total number of positional embeddings for the image (-1 excludes CLS token)

        if npatch == N and h == w:
            return pos_embed

        patch_pos_embed = pos_embed.float()[:, 1:] # Skip CLS token; (1, sqrt(N) * sqrt(N), dim)

        h0 = h // dinov2.patch_size + .1 # Number of patches in height + .1 to avoid rounding errors
        w0 = w // dinov2.patch_size + .1 # Number of patches in width + .1 to avoid rounding errors

        sqrt_N = np.sqrt(N)
        assert np.isclose(sqrt_N, int(sqrt_N)) # N must be a perfect square
        sqrt_N = int(sqrt_N)

        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N # How much to scale the positional embeddings to match w0, h0
        patch_pos_embed = F.interpolate(
            rearrange(patch_pos_embed, 'b (h w) d -> b d h w', h=sqrt_N, w=sqrt_N), # (1, dim, sqrt(N), sqrt(N))
            scale_factor=(sy, sx),
            mode="bicubic",
        ) # (1, dim, h, w)

        patch_pos_embed = rearrange(patch_pos_embed, '1 d h w -> h w d') # Squeeze and move dim to the end

        assert int(h0) == patch_pos_embed.shape[0]
        assert int(w0) == patch_pos_embed.shape[1]

        return patch_pos_embed

    def downsample_sam_masks(self) -> torch.IntTensor:
        '''
        Downsamples the sam_masks for an image to match the dino_feats shape (the patch feature shapes):
        (nmasks, h, w) -> (nmasks, npatches_h, npatches_w)
        '''
        # CenterPadding pads height and width to be a multiple of 14 equally on left and right
        sam_masks = self.padded_sam_masks # (nmasks, padded_h, padded_w)
        sam_masks = rearrange(sam_masks, 'b h w -> b 1 h w') # interpolate needs a channel dimension

        down_sam_mask = F.interpolate(
            sam_masks.contiguous(), # Segfaults if not contiguous
            size=self.dino_feats.shape[1:],
            mode='bilinear'
        ) # (nmasks, 1, npatches_h, npatches_w)

        return down_sam_mask.squeeze()

    def get_region_embeddings(self, down_sam_masks: torch.IntTensor = None, down_pos_embeds: torch.FloatTensor = None) -> torch.FloatTensor:
        '''
        Returns embeddings for the SAM regions as the average of the DinoV2 positional embeddings corresponding to image
        patches containing the downsampled SAM regions.

        Args:
            down_sam_masks (torch.IntTensor, optional): Output of downsample_sam_masks. (nmasks, npatches_h, npatches_w).
            Defaults to None.

            down_pos_embeds (torch.FloatTensor, optional): Output of downsample_positional_embeddings. (npatches_h, npatches_w, dim).
            Defaults to None.

        Returns:
            torch.FloatTensor: (nmasks, dim)
        '''
        if down_sam_masks is None:
            down_sam_masks = self.downsample_sam_masks().int() # Call int in case was cast to float for cuda
        if down_pos_embeds is None:
            down_pos_embeds = self.downsample_positional_embeddings()

        # Extract the region embeddings
        bin_sam_masks = down_sam_masks.to(torch.bool) # Should already be binary, but just in case. (nmasks, npatches_h, npatches_w)

        masked_pos_embeds = down_pos_embeds * bin_sam_masks.unsqueeze(-1) # (nmasks, npatches_h, npatches_w, dim)

        mask_embeds = reduce(masked_pos_embeds, 'n h w d -> n d', 'sum') # (nmasks, dim); sum masked embeddings over patches
        mask_embeds = mask_embeds / reduce(bin_sam_masks, 'n h w -> n', 'sum').unsqueeze(-1) # (nmasks, dim); divide by number of summed nonzero embeddings

        return mask_embeds # (nmasks, dim)

def parse_args(cl_args: List[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', help='Location of extracted DinoV2 features')
    parser.add_argument('--sam_dir', help='Location of SAM masks')
    parser.add_argument('--output_dir', help='Where to save the SAM region embeddings')
    parser.add_argument('--dino_model', default='dinov2_vitl14', help='DinoV2 model used to generate the image features')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use for computing embeddings')

    args = parser.parse_args(cl_args)

    return args

# %%
if __name__ == '__main__':
    args = parse_args([
        '--feature_dir', '/shared/rsaas/dino_sam/features/dinov2/ADE20K/train',
        '--sam_dir', '/shared/rsaas/dino_sam/sam_output/ADE20K/train',
        '--output_dir', '/shared/rsaas/dino_sam/sam_region_embeddings/ADE20K/train',
        '--device', 'cpu'
    ])

    if args.device == 'cuda':
        logger.warning(
            'Cuda interpolate needs floats and outputs a small number of non-binary masks (in one test, .05% were non-binary).'
            ' Unless speed is a necessity, CPU interpolation is recommended.'
        )

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    dino = torch.hub.load('facebookresearch/dinov2', args.dino_model).to(device)
    padder = CenterPadding(multiple=dino.patch_size)

    logger.info(f'Generating SAM region embeddings for {args.sam_dir}')
    for sam_basename in tqdm(os.listdir(args.sam_dir)):
        # Load SAM masks
        sam_path = os.path.join(args.sam_dir, sam_basename)
        with open(sam_path) as f:
            mask_data = json.load(f)

        sam_masks = torch.stack([
            torch.from_numpy(mask_utils.decode(mask['segmentation']))
            for mask in mask_data
        ]).to(device) # (nmasks, h, w)

        if args.device == 'cuda':
            sam_masks = sam_masks.float() # Cuda needs floats to interpolate

        # Load DinoV2 features
        feature_path = os.path.join(args.feature_dir, sam_basename.replace('.json', '.pkl'))

        if not os.path.exists(feature_path):
            logger.warning(f'Feature path {feature_path} does not exist')
            continue

        with open(feature_path, 'rb') as f:
            dino_feats = pickle.load(f)[0] # (d, npatches_h, npatches_w)

        dino_feats = torch.from_numpy(dino_feats).to(device)

        # Embed regions
        generator = RegionEmbeddingGenerator(
            dino_feats=dino_feats,
            sam_masks=sam_masks,
            dinov2=dino,
            padder=padder
        )

        region_embeds = generator.get_region_embeddings().cpu().numpy()

        out_path = os.path.join(args.output_dir, sam_basename.replace('.json', '.pkl'))
        with open(out_path, 'wb') as f:
            pickle.dump(region_embeds, f)

    # %% Testing code
    # dino_dir = '/shared/rsaas/dino_sam/features/dinov2/ADE20K/train'
    # sam_dir = '/shared/rsaas/dino_sam/sam_output/ADE20K/train'
    # ex_name = 'ADE_train_00000001'

    # sam_path = os.path.join(sam_dir, f'{ex_name}.json')
    # dino_feature_path = os.path.join(dino_dir, f'{ex_name}.pkl')

    # with open(dino_feature_path, 'rb') as f:
    #     dino_feats = torch.from_numpy(pickle.load(f)[0]) # (d, npatches_h, npatches_w)

    # with open(sam_path) as f:
    #     mask_data = json.load(f)

    # sam_masks = torch.stack([
    #     torch.from_numpy(mask_utils.decode(mask['segmentation']))
    #     for mask in mask_data
    # ]) # (nmasks, h, w)

    # # %% Load Dinov2
    # dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    # padder = CenterPadding(multiple=dino.patch_size)

    # # %%
    # generator = RegionEmbeddingGenerator(
    #     dino_feats=dino_feats,
    #     sam_masks=sam_masks,
    #     dinov2=dino,
    #     padder=padder
    # )

    # downsampled_masks = generator.downsample_sam_masks()
    # pos_embeds = generator.downsample_positional_embeddings()

    # %%
