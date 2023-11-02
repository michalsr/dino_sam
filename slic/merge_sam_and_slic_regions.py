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
from einops import rearrange, reduce

def get_unmasked_slic_regions(slic_assignment: torch.IntTensor, sam_masks: torch.Tensor, min_proportion: float = 0.):
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
        proportion_in_intersection = (slic_region & no_sam_mask).sum() / slic_region.sum()

        if proportion_in_intersection >= min_proportion:
            slic_regions.append(slic_region)

    return torch.stack(slic_regions) # (n,h,w)

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
    sam_dir = '/shared/rsaas/dino_sam/sam_output/pascal_voc/train'
    slic_dir = '/home/blume5/dino_sam/outputs/slic/pascal/all'
    img_dir = '/shared/rsaas/dino_sam/data/VOCdevkit/VOC2012/JPEGImages'
    min_proportion = 0

    for mask_basename in sorted(os.listdir(sam_dir))[:5]:
        mask_path = os.path.join(sam_dir, mask_basename) # JSON
        slic_path = os.path.join(slic_dir, os.path.splitext(mask_basename)[0] + '.pkl')

        # Load SAM masks
        with open(mask_path, 'r') as f:
            sam_masks = json.load(f)

        sam_masks = torch.tensor(
            np.stack([mask_utils.decode(mask['segmentation']) for mask in sam_masks])
        , dtype=torch.bool) # (n,h,w)

        # Load SLIC regions
        with open(slic_path, 'rb') as f:
            slic_assignment = torch.tensor(pickle.load(f)['assignment'])

        unmasked_slic_regions = get_unmasked_slic_regions(slic_assignment, sam_masks, min_proportion=min_proportion)

        # Load image
        img_path = os.path.join(img_dir, os.path.splitext(mask_basename)[0] + '.jpg')
        with Image.open(img_path) as f:
            img = rearrange(torch.tensor(np.array(f)), 'h w c -> c h w')

        # Visualize SAM masks and SLIC regions
        masked_regions_img = image_from_masks(sam_masks, superimpose_on_image=img) # SAM mask regions
        unmasked_regions_img = sam_masks.any(dim=0).logical_not().int() # Regions unmasked by SAM
        superpixel_assignment_img = img_from_superpixels(img, slic_assignment) # Superpixel assignment
        slic_covering_unmasked_img = image_from_masks(unmasked_slic_regions, superimpose_on_image=img) # SLIC covering unmasked regions
        slic_and_sam_img = image_from_slic_and_sam(img, sam_masks, unmasked_slic_regions) # SLIC covering unmasked regions + SAM masks

        slic_intersect_unmasked = unmasked_slic_regions.any(dim=0) * unmasked_regions_img.bool()
        slic_and_sam_smooth_img = image_from_slic_and_sam(img, sam_masks, slic_intersect_unmasked.unsqueeze(0)) # SLIC intersect unmasked regions + SAM masks

        show(masked_regions_img)
        show(unmasked_regions_img)
        show(superpixel_assignment_img)
        show(slic_covering_unmasked_img)
        show(slic_and_sam_img)
        show(slic_and_sam_smooth_img)

        show([masked_regions_img, unmasked_regions_img, slic_covering_unmasked_img, slic_and_sam_smooth_img], nrows=2)

# %%
