# %%
'''
Script that takes the unlabeled segmentation regions of a dataset and computes the maximum possible
mIoU for that dataset by assigning each region to the label corresponding to the region that it overlaps
with the most, then computing the mIoU of these labeled regions and the ground truth segmentations.
'''
import os
import sys # Path hacks to make imports work
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from utils import open_file, save_file, mean_iou
from typing import List, Dict, Any
import torch
import logging
import coloredlogs
from torchvision.io import read_image
import matplotlib.pyplot as plt
from pycocotools import mask as mask_utils
from einops import rearrange, einsum, reduce
from dataclasses import dataclass
from visualize_sam import get_colors, image_from_masks
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np

logger = logging.getLogger(__name__)
coloredlogs.install(logging.INFO, logger=logger)

@dataclass
class Example:
    sam_output_path: str
    gt_path: str
    img_path: str

@dataclass
class ProcessedExample:
    img_name: str
    sam_img: torch.Tensor
    gt_img: torch.Tensor
    num_in_mult_regions: int
    did_shift_labels: bool

def split_gt_into_masks(gt_img: torch.Tensor, ignore_zero_label: bool):
    labels = torch.tensor(
        [l for l in gt_img.unique() if l != 0 or not ignore_zero_label], # Set of (potentially nonzero) labels
        device=gt_img.device,
    )

    if any(labels < 0):
        raise ValueError('Label value less than zero detected. This is not supported.')

    if len(labels) == 0:
        err_str = 'Failed to split GT image into masks; no labels detected.'
        logger.error(err_str)
        raise RuntimeError(err_str)

    if not ignore_zero_label:
        labels += 1
        gt_img += 1

    gt_masks = torch.stack([(gt_img == l) * l for l in labels]) # (nlabels, h, w)

    return gt_masks, labels

def label_regions(sam_regions: torch.Tensor, gt_img: torch.Tensor, ignore_zero_label: bool = False):
    '''
    Splits the GT segmentations into binary masks for each label, then assigns each SAM region to the label
    corresponding to the GT label with which it has the maximum intersection.

    Arguments:
        sam_regions (torch.Tensor): (nregions, h, w)

        gt_image (torch.Tensor): (h, w)

    Returns:
        labeled_regions (torch.Tensor): The stacked SAM regions, but with each region having the label value corresponding to its best-matched GT label.
            Labels are possibly shifted by 1. (nregions, h, w)

        gt_masks (torch.Tensor): The gt_img, split into masks with the value for each label. Labels are possibly shifted by 1. (nlabels, h, w)
    '''
    gt_masks, labels = split_gt_into_masks(gt_img, ignore_zero_label) # (nlabels, h, w)

    bin_sam_regions = sam_regions.to(torch.bool) # Binary mask
    bin_gt_masks = gt_masks.to(torch.bool) # Binary mask

    intersections = einsum(bin_sam_regions, bin_gt_masks, 'nregions h w , nlabels h w -> nregions nlabels h w')
    intersection_sizes = reduce(intersections, 'nregions nlabels h w -> nregions nlabels', 'sum')

    # NOTE If there is no intersection between a SAM region and any GT region, this implementaiton assigns the SAM region to the 0-index label
    region_to_label_ind = intersection_sizes.argmax(dim=1) # (nregions,)
    region_to_label = labels[region_to_label_ind] # (nregions,)

    labeled_regions = bin_sam_regions * rearrange(region_to_label, 'nregions -> nregions 1 1') # (nregions, h, w)

    return labeled_regions, gt_masks

def num_pixels_in_multiple_regions(sam_segs: torch.Tensor):
    sam_segs = sam_segs.to(torch.bool) # Binary mask
    return (reduce(sam_segs, 'b h w -> h w', 'sum') > 1).sum().item()

def load_sam_regions(sam_output_path: str, gt_shape: torch.Size):
    '''
    Loads the SAM regions from the given path. Tensor of shape (nregions, h, w).

    Args:
        sam_output_path (str): Path to the SAM output JSON.
        gt_shape (torch.Size): Shape of the ground truth segmentation. Used to create empty masks if there are no regions.

    Returns:
        torch.IntTensor: Binary uint8 tensor of shape (nregions, h, w).
    '''
    sam_output: List[Dict[str,Any]] = open_file(sam_output_path)
    sam_regions = [mask_utils.decode(region['segmentation']) for region in sam_output]

    if len(sam_regions) == 0:
        logger.info(f'No regions detected for {sam_output_path}; returning empty mask')
        return torch.zeros(1, *gt_shape, dtype=torch.uint8)

    sam_regions = torch.stack([torch.from_numpy(s) for s in sam_regions])

    return sam_regions

def collapse_masks(stacked_masks: torch.Tensor):
    '''
    Collapses a stack of masks into a single image with each pixel having the label of the region.

    Some pixels may be in multiple regions, and so would have multiple labels. We arbitrarily assign these
    pixels to the last labeled region with which they overlap.

    Args:
        stacked_masks (torch.Tensor): (nregions, h, w)

    Returns:
        torch.Tensor: (h, w) with values equal to the labels.
    '''
    labeled_regions = torch.zeros(stacked_masks.shape[1:], device=stacked_masks.device, dtype=stacked_masks.dtype) # (h,w)
    for region in stacked_masks:
        nonzero_locs = region.nonzero(as_tuple=True)
        labeled_regions[nonzero_locs] = region[nonzero_locs]

    return labeled_regions

def channelize_image(img: torch.Tensor, colors: torch.Tensor):
    '''
    Colors the image whose values are used to select from the given colors.

    Args:
        img (torch.Tensor): (h,w)
        colors (torch.Tensor): (max(img.unique()), 3)

    Returns:
        torch.Tensor: (3, h, w) The image colored with the given colors depending on the label values in img.
    '''
    channelized_img = torch.zeros(*img.shape, 3, device=img.device, dtype=colors.dtype)

    # Ignore the zero-class as it is either shifted to 1, or ignored
    nonzero_locs = img.nonzero(as_tuple=True)
    channelized_img[nonzero_locs[0], nonzero_locs[1], :] = colors[img[nonzero_locs].int()] # Don't index colors with uint8

    return channelized_img

def visualize_labeled_sam_and_gt(
    img: torch.Tensor,
    sam_masks: torch.Tensor,
    gt_masks: torch.Tensor,
    save_file_path: str = None
):
    '''
    Visualize the labeled SAM masks and the ground truth segmentations.

    Args:
        img (torch.Tensor): The original image. (3, h, w)
        sam_masks (torch.Tensor): The labeled sam regions, output by label_regions. (nregions, h, w)
        gt_masks (torch.Tensor): The split gt masks, output by label_regions. (nlabels, h, w)
    '''
    sam_img = collapse_masks(sam_masks) # (h, w)
    gt_img = collapse_masks(gt_masks) # (h, w)

    # Construct the images. For vectorization, we get as many colors as the max label value
    colors = torch.from_numpy(get_colors(gt_img.max().item() + 1)) # +1 to index the max value into the colors tensor

    sam_img = channelize_image(sam_img, colors.to(sam_img.device))
    gt_img = channelize_image(gt_img, colors.to(gt_img.device))

    # Plot the images
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))

    axs[0,0].imshow(rearrange(img, 'c h w -> h w c'))
    axs[0,0].set_title('Original Image')
    axs[0,0].set_xticks([])
    axs[0,0].set_yticks([])

    axs[0,1].imshow(rearrange(image_from_masks(sam_masks.to(bool)), 'c h w -> h w c'))
    axs[0,1].set_title('Unlabeled SAM Regions')
    axs[0,1].set_xticks([])
    axs[0,1].set_yticks([])

    axs[1,1].imshow(sam_img)
    axs[1,1].set_title('Labeled SAM Regions')
    axs[1,1].set_xticks([])
    axs[1,1].set_yticks([])

    axs[1,0].imshow(gt_img)
    axs[1,0].set_title('Ground Truth Segmentations')
    axs[1,0].set_xticks([])
    axs[1,0].set_yticks([])

    if save_file_path:
        fig.savefig(save_file_path)

def compute_iou(pred_imgs: List[np.ndarray], gt_imgs: List[np.ndarray], did_shift_labels: bool, args: argparse.Namespace):
    labels = set()
    for gt in gt_imgs:
        labels.update(np.unique(gt))

    # If we are not ignoring zero, we shifted all the labels by 1. So 0 is not a label.
    # If we are ignoring zero, we didn't shift the labels. But 0 is still not a label since we are ignoring it
    num_classes = len(labels) - 1 if 0 in labels else len(labels)

    miou = mean_iou(
        results=pred_imgs,
        gt_seg_maps=gt_imgs,
        num_labels=num_classes,
        ignore_index=255, # XXX This MUST be set to 255 to work with reduce_labels, since reduce_labels sets the 0 index to 255
        reduce_labels=did_shift_labels, # Downshift labels by 1 if we shifted them up
        reduce_pred_labels=did_shift_labels
    )

    logger.info(miou)

    save_file(os.path.join(args.output_dir, 'mean_iou.json'), miou, json_numpy=True)

def process_example(example: Example, args: argparse.Namespace):
    gt = read_image(example.gt_path).squeeze().to(args.device)
    sam_regions = load_sam_regions(example.sam_output_path, gt.shape).to(args.device)

    try: # May fail to label regions if the GT segmentation is invalid (all zeros and we ignore zeros)
        sam_masks, gt_masks = label_regions(sam_regions, gt, args.ignore_zero)
    except RuntimeError:
        logger.warning(f'Failed to label regions for {example.sam_output_path}; skipping')
        return None

    num_in_mult_regions = num_pixels_in_multiple_regions(sam_regions)

    sam_img = collapse_masks(sam_masks)
    assert (gt == collapse_masks(gt_masks)).all()

    return ProcessedExample(
        img_name=os.path.basename(example.sam_output_path).split('.')[0],
        sam_img=sam_img.cpu().numpy(),
        gt_img=gt.cpu().numpy(),
        num_in_mult_regions=num_in_mult_regions,
        did_shift_labels=not args.ignore_zero,
    )

def gather_examples(sam_dir: str, annotation_dir: str, image_dir: str):
    # Collect paths for SAM outputs and corresponding GT and original images
    num_sam_outputs = len([f for f in os.listdir(sam_dir) if f.endswith('.json')])
    num_gts = len([f for f in os.listdir(annotation_dir) if f.endswith('.png')])

    if num_sam_outputs != num_gts:
        logger.warning(f'Number of SAM outputs ({num_sam_outputs}) does not match number of ground truth segmentations ({num_gts})')

    examples = []
    for sam_basename in tqdm(sorted(os.listdir(sam_dir)), desc='Gathering examples'):
        gt_path = os.path.join(annotation_dir, sam_basename.replace('.json', '.png'))
        sam_path = os.path.join(sam_dir, sam_basename)
        img_path = os.path.join(image_dir, sam_basename.replace('.json', '.jpg'))

        if not os.path.exists(gt_path):
            logger.warning(f'Ground truth segmentation for {sam_path} does not exist; skipping')
            continue

        if not os.path.exists(img_path):
            logger.warning(f'Image for {sam_path} does not exist; skipping')
            continue

        examples.append(Example(sam_output_path=sam_path, gt_path=gt_path, img_path=img_path))

    return examples

def parse_args(cl_args: List[str] = None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--processed_example_dir', help='If provided, loads the processed examples from this directory instead of processing them.')

    parser.add_argument('--sam_dir', help='Directory of generated SAM regions')
    parser.add_argument('--annotation_dir', help='Directory of ground-truth per-pixel annotations')
    parser.add_argument('--image_dir', help='Directory of original images')
    parser.add_argument('--output_dir', help='Directory to output the processed examples with labeled regions.')

    parser.add_argument('--ignore_zero', action='store_true', help='Ignore the 0 label (background)')

    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use for computation')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of labeling jobs to run in parallel')

    return parser.parse_args(cl_args)

# %%
if __name__ == '__main__':
    # TODO move this into main function
    # NOTE GT values indicates LABELS; not regions
    args = parse_args([
        # '--processed_example_dir', './labeled_regions',

        ## ADE20K
        '--sam_dir', '/shared/rsaas/dino_sam/hq_sam_output/ADE20K/train',
        '--annotation_dir', '/shared/rsaas/dino_sam/data/ADE20K/annotations/training',
        '--image_dir', '/shared/rsaas/dino_sam/data/ADE20K/images/training',
        '--output_dir', './ade20k/hq_sam_labeled_regions',
        '--ignore_zero',

        ## PASCAL VOC
        # '--sam_dir', '/shared/rsaas/dino_sam/mobile_sam_output/pascal_voc/train',
        # '--annotation_dir', '/shared/rsaas/dino_sam/data/VOCdevkit/VOC2012/segmentation_annotation/train',
        # '--image_dir', '/shared/rsaas/dino_sam/data/VOCdevkit/VOC2012/segmentation_imgs/train',
        # '--output_dir', './pascal_voc/mobile_sam_labeled_regions',

        '--n_jobs', '10',
        '--device', 'cpu', # cuda runs out of memory after a while; possible memory leak?
    ])

    # %%
    if args.processed_example_dir is None:
        logger.info('Gathering examples...')
        examples = gather_examples(args.sam_dir, args.annotation_dir, args.image_dir)

        # Assign labels to SAM regions
        logger.info('Assigning labels to SAM regions')
        proc_exs = Parallel(n_jobs=args.n_jobs)(delayed(process_example)(example, args) for example in tqdm(examples))
        proc_exs = [ex for ex in proc_exs if ex is not None] # None means failed to label regions

        pct_pixels_in_mult_regions = sum([ex.num_in_mult_regions for ex in proc_exs]) / sum([ex.sam_img.size for ex in proc_exs]) * 100
        logger.info(f'Percentage of pixels in multiple regions across dataset: {pct_pixels_in_mult_regions:.2f}%')

        if args.output_dir:
            logger.info(f'Saving labeled regions to {args.output_dir}')
            os.makedirs(args.output_dir, exist_ok=True)

            for ex in proc_exs:
                save_file(os.path.join(args.output_dir, f'{ex.img_name}.pkl'), ex)

    else:
        proc_exs = [
            open_file(os.path.join(args.processed_example_dir, f))
            for f in os.listdir(args.processed_example_dir)
            if f.endswith('.pkl')
        ]

    # Compute maximum possible mIoU
    logger.info('Computing maximum possible mIoU')
    sam_imgs = [ex.sam_img for ex in proc_exs]
    gt_imgs = [ex.gt_img for ex in proc_exs]
    did_shift_labels = proc_exs[0].did_shift_labels

    compute_iou(sam_imgs, gt_imgs, did_shift_labels, args)
    logger.info('Done!')

    # %% Load example SAM outputs and GTs and visualize the labeling
    # examples = gather_examples(args.sam_dir, args.annotation_dir, args.image_dir)

    # # %%
    # n_examples = 3
    # for example in examples[:n_examples]:
    #     gt = read_image(example.gt_path).squeeze()

    #     sam_regions = load_sam_regions(example.sam_output_path, gt.shape) # (num_regions, H, W)
    #     num_in_mult_regions = num_pixels_in_multiple_regions(sam_regions)

    #     sam_masks, gt_masks = label_regions(sam_regions, gt, args.ignore_zero)

    #     img_name = os.path.basename(example.sam_output_path).split('.')[0]
    #     visualize_labeled_sam_and_gt(
    #         read_image(example.img_path),
    #         sam_masks,
    #         gt_masks,
    #         save_file_path=f'{img_name}.png'
    #     )

# %%
