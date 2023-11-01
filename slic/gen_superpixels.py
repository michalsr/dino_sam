# %%
'''
Script to generate SLIC superpixels for an image dataset.
'''
import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import pickle
import numpy as np
from PIL import Image
from fast_slic import Slic
from tqdm import tqdm
import torch
from sam_analysis.visualize_sam import image_from_masks, show, masks_to_boundaries

def img_from_superpixels(img: torch.tensor, assignment: torch.tensor):
    # On top of the image
    regions = torch.stack([
        assignment == v
        for v in np.unique(assignment)
    ])

    boundaries = masks_to_boundaries(regions)

    overlaid_img = image_from_masks(boundaries, combine_as_binary_mask=True, superimpose_on_image=img)

    return overlaid_img

# %%
if __name__ == '__main__':
    # Define the input and output directories
    # input_dir = '/shared/rsaas/dino_sam/data/ADE20K/images/training' # Image directory
    input_dir = '/shared/rsaas/dino_sam/data/VOCdevkit/VOC2012/JPEGImages'
    output_dir = '/home/blume5/dino_sam/outputs/slic/pascal/all' # Segmentation data output directory

    # Define the parameters for the superpixel algorithm
    num_components = 1000
    compactness = 10

    os.makedirs(output_dir, exist_ok=True)

    # Loop through each image in the input directory
    basenames = sorted(os.listdir(input_dir))
    # basenames = sorted(os.listdir(input_dir))[:10]

    for filename in tqdm(basenames):
        # Load the image
        with Image.open(os.path.join(input_dir, filename)) as f:
            img = np.array(f)

        # Apply the superpixel algorithm
        slic = Slic(num_components=num_components, compactness=compactness)
        assignment = slic.iterate(img)

        # Save superpixel data
        ret_dict = {
            'assignment': assignment,
            'clusters': slic.slic_model.clusters
        }

        output_filename = os.path.join(output_dir, os.path.splitext(filename)[0] + '.pkl')
        with open(output_filename, 'wb') as f:
            pickle.dump(ret_dict, f)

        # %% Visualize
        # vis_device = 'cpu' # Device for visualization; SLIC runs on the CPU

        # assignment = torch.tensor(assignment, device=vis_device)
        # img = torch.tensor(img.transpose(2, 0, 1), device=vis_device) # HWC -> CHW
        # overlaid_img = img_from_superpixels(img, assignment)
        # show(overlaid_img)
# %%
