# %%
'''
Script to find SLIC superpixels which have nonempty intersection with unmasked
regions of an image (i.e. regions which are not masked out by the SAM segmentation).
'''
import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import json
import pickle
from PIL import Image
from pycocotools import mask as mask_utils
from gen_superpixels import img_from_superpixels
from sam_analysis.visualize_sam import image_from_masks, show, masks_to_boundaries
from tqdm import tqdm
from einops import rearrange
from to_sam_format import stacked_masks_to_sam_dicts

import coloredlogs, logging
logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', logger=logger)

def get_unmasked_slic_regions(slic_assignment: torch.IntTensor, sam_masks: torch.BoolTensor, min_proportion: float = 0., min_pixels: int = 0) -> torch.BoolTensor:
    '''
    Get the SLIC regions which have nonempty intersection with unmasked regions of an image.

    Args:
        slic_assignment (torch.IntTensor): Tensor of SLIC superpixel assignments. (h,w).
        sam_masks (torch.IntTensor): Binary tensor of stacked SAM masks. (n,h,w).
        min_proportion (float): Minimum proportion of a SLIC region which must be in the intersection

    Returns:
        slic_regions (torch.Tensor): Binary tensor of SLIC regions with nonzero intersection. (n,h,w).
    '''
    # Collapse SAM masks into a single binary mask indicating where regions are not
    any_sam_mask = sam_masks.any(dim=0) # (h,w)
    no_sam_mask = ~any_sam_mask # (h,w)

    slic_vals_in_no_sam_mask = set(slic_assignment[no_sam_mask].tolist())

    # Keep only the SLIC regions which have a minimum proportion of their pixels in the intersection
    slic_regions = []
    for val in slic_vals_in_no_sam_mask:
        slic_region = slic_assignment == val
        intersection = slic_region & no_sam_mask

        n_pixels_in_intersection = intersection.sum()
        proportion_in_intersection = n_pixels_in_intersection / slic_region.sum()

        if n_pixels_in_intersection >= min_pixels and proportion_in_intersection >= min_proportion:
            slic_regions.append(slic_region)

    if len(slic_regions) == 0:
        return torch.zeros_like(slic_assignment).unsqueeze(0).bool() # (1,h,w)

    return torch.stack(slic_regions) # (n,h,w)

def get_merged_sam_and_slic_regions(unmasked_slic_regions: torch.BoolTensor, sam_masks: torch.BoolTensor):
    '''
    Merge the SLIC regions which have nonempty intersection with unmasked regions of an image, then stack them with the
    SAM masks.
    '''
    # Check if all SLIC regions got filtered out
    if len(unmasked_slic_regions) == 1 and not unmasked_slic_regions.any():
        return sam_masks

    intersected_slic_regions = (unmasked_slic_regions * sam_masks.any(dim=0).logical_not()).bool() # Intersect with unmasked regions; (n_slic,h,w)

    return torch.cat([sam_masks, intersected_slic_regions]) # (n_slic + n_sam,h,w)

def image_from_slic_and_sam(img: torch.Tensor, sam_masks, slic_regions: torch.Tensor, show_boundaries=False):
    '''
    Args:
        img (torch.Tensor): (c, h, w)
        slic_regions (torch.Tensor): (n, h, w)
    '''
    # Visualize the SAM masks
    img = image_from_masks(sam_masks, superimpose_on_image=img)

    # Number the regions and collapse
    if show_boundaries:
        slic_regions = masks_to_boundaries(slic_regions)

    return image_from_masks(slic_regions, combine_as_binary_mask=True, superimpose_on_image=img)

# %%
if __name__ == '__main__':
    ## Pascal
    sam_dir = '/shared/rsaas/dino_sam/sam_output/pascal_voc/train'
    slic_dir = '/home/blume5/shared/slic_50_8/pascal_voc/assignments/train'
    img_dir = '/shared/rsaas/dino_sam/data/VOCdevkit/VOC2012/JPEGImages'

    ## ADE20K
    # sam_dir = '/shared/rsaas/dino_sam/sam_output/ADE20K/train' # SAM masks to merge with the SLIC regions
    # slic_dir = '/home/blume5/shared/slic_50_8/ade20k/assignments/train' # SLIC regions to merge with the SAM masks
    # img_dir = '/shared/rsaas/dino_sam/data/ADE20K/images/training' # Images to visualize the merged regions on

    sam_output_dir = '/home/blume5/shared/sam_slic_50_8/ade20k/sam_regions/train' # Where to save the SAM + SLIC output as SAM output JSONs

    min_proportion = 0
    min_pixels = 300

    device = 'cpu'

    os.makedirs(sam_output_dir, exist_ok=True)
    for mask_basename in tqdm(sorted(os.listdir(sam_dir))[11:12]):
        mask_path = os.path.join(sam_dir, mask_basename) # JSON
        slic_path = os.path.join(slic_dir, os.path.splitext(mask_basename)[0] + '.pkl')

        if not os.path.exists(slic_path):
            logger.warning(f'No SLIC assignment found at {slic_path}. Skipping.')
            continue

        # Load SAM masks
        with open(mask_path, 'r') as f:
            sam_masks = json.load(f)

        sam_masks = torch.tensor(
            np.stack([mask_utils.decode(mask['segmentation']) for mask in sam_masks])
        , dtype=torch.bool, device=device) # (n,h,w)

        # Load SLIC regions
        with open(slic_path, 'rb') as f:
            slic_assignment = torch.tensor(pickle.load(f)['assignment'], device=device)

        # Merge SLIC with SAM
        unmasked_slic_regions = get_unmasked_slic_regions(slic_assignment, sam_masks, min_proportion=min_proportion, min_pixels=min_pixels)
        merged_slic_and_sam_regions = get_merged_sam_and_slic_regions(unmasked_slic_regions, sam_masks).cpu()

        # Dump merged regions to SAM output file
        # image_id = os.path.splitext(mask_basename)[0]
        # sam_dicts = stacked_masks_to_sam_dicts(merged_slic_and_sam_regions.numpy(), image_id)

        # with open(os.path.join(sam_output_dir, f'{image_id}.json'), 'w') as f:
        #     json.dump(sam_dicts, f)

        # Visualize SAM masks and SLIC regions
        # Load image
        img_path = os.path.join(img_dir, os.path.splitext(mask_basename)[0] + '.jpg')
        with Image.open(img_path) as f:
            img = rearrange(torch.tensor(np.array(f)), 'h w c -> c h w')

        masked_regions_img = image_from_masks(sam_masks, superimpose_on_image=img) # SAM mask regions
        unmasked_regions_img = sam_masks.any(dim=0).logical_not().int() # Regions unmasked by SAM
        superpixel_assignment_img = img_from_superpixels(img, slic_assignment) # Superpixel assignment
        slic_covering_unmasked_img = image_from_masks(unmasked_slic_regions, superimpose_on_image=img) # SLIC covering unmasked regions
        slic_and_sam_img = image_from_slic_and_sam(img, sam_masks, unmasked_slic_regions) # SLIC covering unmasked regions + SAM masks

        slic_intersect_unmasked = unmasked_slic_regions.any(dim=0) * unmasked_regions_img.bool()
        slic_and_sam_smooth_img = image_from_slic_and_sam(img, sam_masks, slic_intersect_unmasked.unsqueeze(0)) # SLIC intersect unmasked regions + SAM masks

        slic_and_sam_smooth_split_img = image_from_masks(merged_slic_and_sam_regions, superimpose_on_image=img) # SLIC intersect unmasked regions, split into superpixels, + SAM masks
        remaining_unmasked = merged_slic_and_sam_regions.any(dim=0).logical_not().int()

        # show(masked_regions_img)
        # show(unmasked_regions_img)
        # show(superpixel_assignment_img)
        # show(slic_covering_unmasked_img)
        # show(slic_and_sam_img)
        # show(slic_and_sam_smooth_img)

        # show(slic_and_sam_smooth_split_img)
        # show(remaining_unmasked)

        show(
            [img, unmasked_regions_img, superpixel_assignment_img, slic_covering_unmasked_img, slic_and_sam_smooth_split_img, remaining_unmasked],
            subplot_titles=[
                'Original image',
                'Unmasked regions',
                'Superpixel assignment',
                'Superpixels covering\nunmasked regions',
                'Full Segmentation\n(SLIC + SAM)',
                'Remaining unmasked'
            ],
            nrows=2,
            fig_kwargs={'figsize': (15, 8.5)}
        )

    # %% Visualize image PCA features and overlay with original image
    img_id = os.path.basename(img_path).replace('.jpg', '')
    feat_path = f'/shared/rsaas/dino_sam/features/dinov2/pascal_voc_layer_23/train/{img_id}.pkl'
    with open(feat_path, 'rb') as f:
        feats = torch.from_numpy(pickle.load(f))

    # %%
    from sklearn.decomposition import PCA
    import torch.nn.functional as F

    upsampled_feats = F.interpolate(feats, size=img.shape[-2:], mode='bilinear')
    pca = PCA(n_components=3)
    points = pca.fit_transform(rearrange(upsampled_feats, '1 c h w -> (h w) c').numpy())

    # %%
    from scipy.ndimage import gaussian_filter

    pca_img = rearrange(points, '(h w) c -> c h w', h=img.shape[-2], w=img.shape[-1])

    # pca_img = gaussian_filter(pca_img, sigma=.5)

    pca_img = torch.tensor(pca_img)
    pca_img -= pca_img.min() # Normalize to [0, 1]
    pca_img /= pca_img.max() # (3, h, w)
    pca_img = (pca_img * 255).to(torch.uint8)
    show(pca_img)

    # %% Overlay the PCA image on top of the SLIC and SAM Sooth Split Image translucently
    pca_img_weight = .7
    blended = pca_img * pca_img_weight + slic_and_sam_smooth_split_img * (1 - pca_img_weight)
    blended = blended.to(torch.uint8)
    show(blended)

    print('Done.')

# %%
