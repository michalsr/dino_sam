'''
Converts SLIC assignment matrices (or stacked masks from any source) to pseudo-SAM dicts usable by process_regions.py.
'''
# %%
import torch
import json
import pickle
import os
from pycocotools import mask as mask_utils
import numpy as np

def stacked_masks_to_sam_dicts(stacked_masks: np.ndarray, image_id: str):
    '''
    Args:
        stacked_masks (np.ndarray): Binary tensor of stacked SAM masks. (n,h,w).
        image_id (str): Image ID of the image the SAM masks came from.
    Returns:
        sam_dicts (list): List of dicts as if output by sam.py with the properties essential for process_regions.py.
    '''
    # Following process_regions.py, we need the following fields for each split mask:
    # region_id: {image_id}_region_{i} from sam.py
    # area: area of the region (sum of binary mask)
    # segmentation: {
    #  "size": [h, w],
    # "counts": RLE-encoded binary mask
    # }

    return [
        {
            'region_id': f'{image_id}_region_{i}',
            'area': mask.sum(),
            'segmentation': mask_utils.encode(mask)
        } for i, mask in enumerate(stacked_masks)
    ]

def assignment_to_sam_dict(assignment: np.ndarray, image_id: str):
    '''Creates a SAM dict from a SLIC image assignment matrix.

    Args:
        assignment (np.ndarray): Int array of SLIC superpixel assignments. Every pixel has an int assignment from 0 to n_regions. (h,w)
        image_id (str): Image ID of the image the SLIC assignment came from.

    Returns:
        sam_dict (list): List of dicts as if output by sam.py with the properties essential for process_regions.py.
    '''
    masks = np.stack([
        assignment == val
        for val in np.unique(assignment)
    ])

    masks = np.asfortranarray(masks.astype(np.uint8))

    return stacked_masks_to_sam_dicts(np.stack(masks), image_id)

# %%
if __name__ == '__main__':
    slic_path = '/home/blume5/shared/slic/ade20k/train/50_8/ADE_train_00000001.pkl'

    with open(slic_path, 'rb') as f:
        assignment = pickle.load(f)['assignment']

    image_id = os.path.splitext(os.path.basename(slic_path))[0]
    dicts = assignment_to_sam_dict(assignment, image_id)
# %%
