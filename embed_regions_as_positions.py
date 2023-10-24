'''
Generates embeddings for SAM regions as the average of the positional embeddings of the patches
containing the downsampled SAM regions.
'''
# %%
import os
import sys
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
from enum import Enum
from joblib import Parallel, delayed
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

class ScalingMethod(Enum):
    UPSCALE_POS_EMBEDS = 'upscale_pos_embeds'
    DOWNSCALE_SAM_MASKS = 'downscale_sam_masks'

class RegionEmbeddingGenerator:
    def __init__(
        self,
        sam_masks: torch.IntTensor,
        dinov2: torch.nn.Module,
        padder: CenterPadding,
        dino_feats: torch.FloatTensor = None,
        scaling_method: ScalingMethod = ScalingMethod.DOWNSCALE_SAM_MASKS
    ):
        '''
        Args:
            sam_masks (torch.Tensor): (nmasks, h, w),
            dinov2 (torch.nn.Module): Dinov2 model used to generate the features.
            padder (torch.nn.Module): Padding module used for the corresponding Dinov2 model.
            dino_feats (torch.Tensor): (d, npatches_h, npatches_w). Necessary if sacling_method is DOWNSCALE_SAM_MASKS.
            scaling_method (ScalingMethod, optional): Method for scaling to match the positional embedding size to the image size.
        '''
        self.sam_masks = sam_masks
        self.dinov2 = dinov2
        self.padder = padder
        self.dino_feats = dino_feats
        self.scaling_method = scaling_method

        # Ensure scaling method is valid
        if not isinstance(scaling_method, ScalingMethod):
            raise ValueError(f'Invalid scaling method {scaling_method}. Must be one of {list(ScalingMethod)}')

        if scaling_method == ScalingMethod.DOWNSCALE_SAM_MASKS:
            if dino_feats is None:
                raise ValueError('dino_feats must be provided if scaling_method is DOWNSCALE_SAM_MASKS')

            # out-channels of the patch convolution is the feature dimension
            if dino_feats.shape[0] != dinov2.patch_embed.proj.out_channels:
                raise ValueError('Feature dimension of dino_feats must match out_channels of Dinov2 patch_embed')

        # CenterPadding expects (b, c, h, w), so add a channel dimension then squeeze it out
        self.padded_sam_masks = padder(sam_masks.unsqueeze(1)).squeeze() # (nmasks, padded_h, padded_w)

    def resample_positional_embeddings(self) -> torch.FloatTensor:
        '''
        Rescales DinoV2's positional embeddings.

        If scaling_method is DOWNSCALE_SAM_MASKS, rescales the positional embeddings to match the dimension of the patch features.
        If scaling_method is UPSCALE_POS_EMBEDS, rescales the positional embeddings to match the dimension of the SAM masks.

        Based on interpolate_pos_encoding and prepare_tokens_with_masks of DinoVisionTransformer class

        Returns:
            torch.FloatTensor: (npatches_h, npatches_w, dim) if scaling_method is DOWNSCALE_SAM_MASKS, else (h, w, dim)
        '''
        # Note that codebase names tensor dimensions as (B, nc, w, h), but based on the output Dino feature dimensions and the
        # width and height of the original image, we believe w h > h w is a naming convention, and it is actually trained for h w.
        h, w = self.padded_sam_masks.shape[1:]

        pos_embed = self.dinov2.pos_embed.data # Extract tensor from parameter

        N = pos_embed.shape[1] - 1 # Total number of positional embeddings for the image (-1 excludes CLS token)
        sqrt_N = np.sqrt(N)
        assert np.isclose(sqrt_N, int(sqrt_N)) # N must be a perfect square
        sqrt_N = int(sqrt_N)

        patch_pos_embed = pos_embed[:, 1:] # Skip CLS token; (1, sqrt(N) * sqrt(N), dim)

        if self.scaling_method == ScalingMethod.DOWNSCALE_SAM_MASKS:
            npatch = self.dino_feats.shape[1] * self.dino_feats.shape[2] # Total number of patches
            if npatch == N and h == w: # Don't need to interpolate as we're already at the right size
                return rearrange(patch_pos_embed, '1 (h w) d -> h w d', h=sqrt_N, w=sqrt_N) # (1, N, dim) -> (sqrt(N), sqrt(N), dim)

            # Compute target pos embed dimensions; scale to match the number of patches
            h0 = h // self.dinov2.patch_size + .1 # Number of patches in height + .1 to avoid rounding errors
            w0 = w // self.dinov2.patch_size + .1 # Number of patches in width + .1 to avoid rounding errors

        elif self.scaling_method == ScalingMethod.UPSCALE_POS_EMBEDS:
            h0, w0 = h + .1, w + .1 # Target pos embed dimensions; upscaling to full mask size

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

    def resample_sam_masks(self) -> torch.IntTensor:
        '''
        If scaling_method is DOWNSCALE_SAM_MASKS, downsamples the sam_masks for an image to match the dino_feats shape
        (the patch feature shapes): (nmasks, h, w) -> (nmasks, npatches_h, npatches_w)

        If scaling method is UPSCALE_POS_EMBEDS, this is a no-op.
        '''
        if self.scaling_method == ScalingMethod.UPSCALE_POS_EMBEDS:
            return self.padded_sam_masks

        elif self.scaling_method == ScalingMethod.DOWNSCALE_SAM_MASKS:
            # CenterPadding pads height and width to be a multiple of 14 equally on left and right
            sam_masks = self.padded_sam_masks # (nmasks, padded_h, padded_w)
            sam_masks = rearrange(sam_masks, 'b h w -> b 1 h w') # interpolate needs a channel dimension

            resampled_sam_mask = F.interpolate(
                sam_masks.contiguous(), # Segfaults if not contiguous
                size=self.dino_feats.shape[1:],
                mode='bilinear'
            ) # (nmasks, 1, npatches_h, npatches_w)

            return resampled_sam_mask.squeeze()

        else:
            raise NotImplementedError(f'Scaling method {self.scaling_method} not implemented')

    def get_region_embeddings(
            self,
            resampled_sam_masks: torch.IntTensor = None,
            resampled_pos_embeds: torch.FloatTensor = None,
            chunk_size: int = 50
        ) -> torch.FloatTensor:
        '''
        Returns embeddings for the SAM regions as the average of the DinoV2 positional embeddings for that region.

        If scaling_method is DOWNSCALE_SAM_MASKS, the positional embeddings are averaged over the downsampled SAM regions.
        If scaling_method is UPSCALE_POS_EMBEDS, the positional embeddings are upscaled and averaged over the full SAM regions.

        Args:
            resampled_sam_masks (torch.IntTensor, optional): Output of resample_sam_masks. (nmasks, npatches_h, npatches_w).
            Defaults to None.

            resampled_pos_embeds (torch.FloatTensor, optional): Output of resample_positional_embeddings. (npatches_h, npatches_w, dim).
            Defaults to None.

        Returns:
            torch.FloatTensor: (nmasks, dim)
        '''
        if resampled_sam_masks is None:
            resampled_sam_masks = self.resample_sam_masks().int() # Call int in case was cast to float for cuda
        if resampled_pos_embeds is None:
            resampled_pos_embeds = self.resample_positional_embeddings()

        # Extract the region embeddings
        bin_sam_masks = resampled_sam_masks.to(torch.bool) # Should already be binary, but just in case. (nmasks, npatches_h, npatches_w)
        bin_sam_mask_chunks = torch.split(bin_sam_masks, chunk_size, dim=0)

        all_mask_embeds = []
        for bin_sam_mask_chunk in bin_sam_mask_chunks:
            masked_pos_embeds_chunk = resampled_pos_embeds * bin_sam_mask_chunk.unsqueeze(-1) # (nmasks_or_chunk_size, npatches_h, npatches_w, dim)
            mask_embeds_chunk = reduce(masked_pos_embeds_chunk, 'n h w d -> n d', 'sum') # (nmasks_or_chunk_size, dim); sum masked embeddings over patches
            mask_embeds_chunk = mask_embeds_chunk / reduce(bin_sam_mask_chunk, 'n h w -> n', 'sum').unsqueeze(-1) # (nmasks_or_chunk_size, dim); divide by number of summed nonzero embeddings
            all_mask_embeds.append(mask_embeds_chunk)

        return torch.cat(all_mask_embeds, dim=0) # (nmasks, dim)

def parse_args(cl_args: List[str] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', help='Location of extracted DinoV2 features')
    parser.add_argument('--sam_dir', help='Location of SAM masks')
    parser.add_argument('--output_dir', help='Where to save the SAM region embeddings')
    parser.add_argument('--dino_model', default='dinov2_vitl14', help='DinoV2 model used to generate the image features')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use for computing embeddings')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs to run in parallel')
    parser.add_argument('--chunk_size', type=int, default=50,
                        help='Number of SAM masks to compute embeddings for at a time. Runs out of memory on CPU if too large and scaling_method is upscale.')

    parser.add_argument('--scaling_method', choices=['upscale_pos_embeds', 'downscale_sam_masks'], default='downscale_sam_masks',
                        help='Method for scaling to match the positional embedding size to the image size')

    args = parser.parse_args(cl_args)

    return args

def gen_embeddings_for_masks(sam_basename: str, dino, padder, args: argparse.Namespace):
    # Load SAM masks
    sam_path = os.path.join(args.sam_dir, sam_basename)
    with open(sam_path) as f:
        mask_data = json.load(f)

    sam_masks = torch.stack([
        torch.from_numpy(mask_utils.decode(mask['segmentation']))
        for mask in mask_data
    ]).to(args.device) # (nmasks, h, w)

    if args.device == 'cuda':
        sam_masks = sam_masks.float() # Cuda needs floats to interpolate

    # Scaling method-specific arguments
    if args.scaling_method == ScalingMethod.DOWNSCALE_SAM_MASKS.value:
        # Load DinoV2 features
        feature_path = os.path.join(args.feature_dir, sam_basename.replace('.json', '.pkl'))

        if not os.path.exists(feature_path):
            logger.warning(f'Feature path {feature_path} does not exist')
            return

        with open(feature_path, 'rb') as f:
            dino_feats = pickle.load(f)[0] # (d, npatches_h, npatches_w)

        dino_feats = torch.from_numpy(dino_feats).to(args.device)
        scaling_method = ScalingMethod.DOWNSCALE_SAM_MASKS

    elif args.scaling_method == ScalingMethod.UPSCALE_POS_EMBEDS.value:
        dino_feats = None
        scaling_method = ScalingMethod.UPSCALE_POS_EMBEDS

    else:
        raise NotImplementedError(f'Scaling method {args.scaling_method} not implemented')

    # Embed regions
    generator = RegionEmbeddingGenerator(
        sam_masks=sam_masks,
        dinov2=dino,
        padder=padder,
        dino_feats=dino_feats,
        scaling_method=scaling_method
    )

    try:
        region_embeds = generator.get_region_embeddings(chunk_size=args.chunk_size).cpu().numpy()

    except Exception as e:
        logger.error(f'Failed to generate region embeddings for path {sam_path}')
        logger.exception(e)
        sys.exit(1)

    out_path = os.path.join(args.output_dir, sam_basename.replace('.json', '.pkl'))
    with open(out_path, 'wb') as f:
        pickle.dump(region_embeds, f)

# %%
if __name__ == '__main__':
    args = parse_args([
        '--feature_dir', '/shared/rsaas/dino_sam/features/dinov2/pascal_voc_layer_23/train',
        '--sam_dir', '/shared/rsaas/dino_sam/sam_output/pascal_voc/train',
        '--output_dir', '/shared/rsaas/dino_sam/sam_region_embeddings/pascal_voc/train_upscale',
        '--device', 'cuda',
        '--n_jobs', '1',
        '--chunk_size', '10',
        '--scaling_method', 'upscale_pos_embeds'
    ])

    if args.device == 'cuda':
        logger.warning(
            'Cuda interpolate needs floats and outputs a small number of non-binary masks (in one test, .05% were non-binary).'
            ' Unless speed is a necessity, CPU interpolation is recommended.'
        )

        if args.n_jobs > 1:
            logger.warning('Parallelization is not supported on cuda')

    os.makedirs(args.output_dir, exist_ok=True)

    dino = torch.hub.load('facebookresearch/dinov2', args.dino_model).to(args.device)
    padder = CenterPadding(multiple=dino.patch_size)

    logger.info(f'Generating SAM region embeddings for {args.sam_dir}')

    if args.n_jobs > 1:
        Parallel(n_jobs=args.n_jobs, backend='threading')(
            delayed(gen_embeddings_for_masks)(sam_basename, dino, padder, args)
            for sam_basename in tqdm(os.listdir(args.sam_dir))
        )

    else:
        for sam_basename in tqdm(os.listdir(args.sam_dir)):
            gen_embeddings_for_masks(sam_basename, dino, padder, args)

    # %% Testing code
    # dino_dir = '/shared/rsaas/dino_sam/features/dinov2/ADE20K/train'
    # sam_dir = '/shared/rsaas/dino_sam/sam_output/ADE20K/train'
    # ex_name = 'ADE_train_00003513'
    # scaling_method = ScalingMethod.UPSCALE_POS_EMBEDS.value

    # sam_path = os.path.join(sam_dir, f'{ex_name}.json')
    # dino_feature_path = os.path.join(dino_dir, f'{ex_name}.pkl')

    # # Handle different scaling methods
    # if scaling_method == ScalingMethod.DOWNSCALE_SAM_MASKS.value:
    #     with open(dino_feature_path, 'rb') as f:
    #         dino_feats = torch.from_numpy(pickle.load(f)[0]) # (d, npatches_h, npatches_w)

    #     scaling_method = ScalingMethod.DOWNSCALE_SAM_MASKS

    # elif scaling_method == ScalingMethod.UPSCALE_POS_EMBEDS.value:
    #     dino_feats = None
    #     scaling_method = ScalingMethod.UPSCALE_POS_EMBEDS

    # else:
    #     raise NotImplementedError(f'Scaling method {scaling_method} not implemented')

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
    #     sam_masks=sam_masks,
    #     dinov2=dino,
    #     padder=padder,
    #     dino_feats=dino_feats,
    #     scaling_method=scaling_method
    # )

    # resampled_masks = generator.resample_sam_masks()
    # pos_embeds = generator.resample_positional_embeddings()
    # region_embeds = generator.get_region_embeddings(resampled_sam_masks=resampled_masks, resampled_pos_embeds=pos_embeds)

    # %%
