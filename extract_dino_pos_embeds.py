# %%
import os
import pickle
import json
from pycocotools import mask as mask_utils
from einops import rearrange
import torch
import numpy as np
import torch
import torch.nn.functional as F
import pickle
import itertools
import math

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

    def create_positional_embeddings(self):
        # Based on interpolate_pos_encoding and prepare_tokens_with_masks of DinoVisionTransformer class
        # Note that codebase names tensor dimensions as (B, nc, w, h), but based on the output Dino feature dimensions and the
        # width and height of the original image, we believe w h > h w is a naming convention, and it is actually trained for h w.
        dino_feats = self.dino_feats
        dinov2 = self.dinov2
        sam_masks = self.padded_sam_masks

        npatch = dino_feats.shape[1] * dino_feats.shape[2] # Total number of patches
        h, w = sam_masks.shape[1:]

        N = dinov2.pos_embed.shape[1] - 1
        if npatch == N and h == w:
            return dinov2.pos_embed

        patch_pos_embed = dinov2.pos_embed.float()[:, 1:] # Skip CLS token; (1, sqrt(N) * sqrt(N), dim)

        h0 = h // dinov2.patch_size + .1 # Number of patches in height + .1 to avoid rounding errors
        w0 = w // dinov2.patch_size + .1 # Number of patches in width + .1 to avoid rounding errors

        print('h0, w0:', h0, w0)

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

    def downsample_sam_mask(self):
        '''
        Downsamples the sam_masks for an image to match the dino_feats shape (the patch feature shapes):
        (nmasks, h, w) -> (nmasks, npatches_h, npatches_w
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

# %%
if __name__ == '__main__':
    dino_dir = '/shared/rsaas/dino_sam/features/dinov2/ADE20K/train'
    sam_dir = '/shared/rsaas/dino_sam/sam_output/ADE20K/train'
    ex_name = 'ADE_train_00000001'

    sam_path = os.path.join(sam_dir, f'{ex_name}.json')
    dino_feature_path = os.path.join(dino_dir, f'{ex_name}.pkl')

    with open(dino_feature_path, 'rb') as f:
        dino_feats = pickle.load(f)[0] # (d, npatches_h, npatches_w)

    with open(sam_path) as f:
        mask_data = json.load(f)

    sam_masks = torch.stack([
        torch.from_numpy(mask_utils.decode(mask['segmentation']))
        for mask in mask_data
    ]) # (nmasks, h, w)

    # %% Load Dinov2
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    padder = CenterPadding(multiple=dino.patch_size)

    # %%
    generator = RegionEmbeddingGenerator(
        dino_feats=dino_feats,
        sam_masks=sam_masks,
        dinov2=dino,
        padder=padder
    )

    downsampled_masks = generator.downsample_sam_mask()
    pos_embeds = generator.create_positional_embeddings()


# %%
