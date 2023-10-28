# %%
import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import json
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from embed_regions_as_positions import CenterPadding
from pycocotools import mask as mask_utils
from PIL import Image
from einops import rearrange
from visualize_sam import image_from_masks, show
import torch.nn.functional as F

def interpolate_pos_embeds(target_h, target_w, pos_embeds: torch.Tensor):
    N = pos_embeds.shape[1] - 1

    pos_embed = pos_embeds.float()
    patch_pos_embed = pos_embed[:, 1:]
    h0 = target_h
    w0 = target_w
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1

    sqrt_N = np.sqrt(N)
    sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), -1).permute(0, 3, 1, 2),
        scale_factor=(sy, sx),
        mode="bicubic",
    )

    assert int(w0) == patch_pos_embed.shape[-1]
    assert int(h0) == patch_pos_embed.shape[-2]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1) # (b, h, w, d)

    return patch_pos_embed

if __name__ == '__main__':
    region_embed_dir = '/shared/rsaas/dino_sam/sam_region_embeddings/ADE20K/train_upscale'
    mask_dir = '/shared/rsaas/dino_sam/sam_output/ADE20K/train'
    image_dir = '/shared/rsaas/dino_sam/data/ADE20K/images/training'
    device = 'cuda'

    ims_to_display = 3
    masks_to_display = 3

    # %% Load positional embeddings
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    padder = CenterPadding(multiple=dino.patch_size)
    pos_embeds = dino.pos_embed.detach().to(device)

    # %%
    for region_embed_basename in sorted(os.listdir(region_embed_dir))[:ims_to_display]:
        # Load region embeds
        region_path = os.path.join(region_embed_dir, region_embed_basename)
        with open(region_path, 'rb') as f:
            region_embeds = torch.from_numpy(pickle.load(f)).to(device)

        # Load masks
        masks_path = os.path.join(mask_dir, region_embed_basename.replace('.pkl', '.json'))
        with open(masks_path, 'rb') as f:
            mask_data = json.load(f)

        masks = torch.stack([
            torch.from_numpy(mask_utils.decode(mask['segmentation']))
            for mask in mask_data
        ]).to(device) # (n_to_display, h, w)

        masks = padder(masks.unsqueeze(0)).squeeze(0) # Expects batch and channel dims

        # Load image
        img_path = os.path.join(image_dir, region_embed_basename.replace('.pkl', '.jpg'))
        img = rearrange(np.array(Image.open(img_path)), 'h w c -> 1 c h w')
        img = padder(torch.from_numpy(img)) # Pad image
        img = rearrange(img, '1 c h w -> c h w') # Squeeze out batch dim

        # Scale pos_embeds
        scaled_pos_embeds = interpolate_pos_embeds(img.shape[1], img.shape[2], pos_embeds)

        for i in range(masks_to_display):
            # Plot mask on image
            mask_img = image_from_masks(masks[i:i+1].bool(), superimpose_on_image=img)
            show(mask_img)

            # Compute positional embedding similarities
            region_embed = region_embeds[None,None,None,i]
            sims = F.cosine_similarity(region_embed, scaled_pos_embeds, 3) # (1, h, w)

            # Map sims to [0, 255]
            sims = (sims + 1) / 2 * 255 # (1, h, w)

            # Heatmap of similarities
            fig, ax = plt.subplots()
            ax.imshow(sims[0].cpu().numpy())
            ax.set_xticks([])
            ax.set_yticks([])

# %%
