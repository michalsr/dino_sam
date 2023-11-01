# %%
'''
Script to generate SLIC superpixels for an image dataset.
'''
import os
import sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from PIL import Image
from fast_slic import Slic
from tqdm import tqdm
import torch
from sam_analysis.visualize_sam import image_from_masks, show, masks_to_boundaries

def visualize_superpixels(img: torch.tensor, assignment: torch.tensor):
    # On top of the image
    regions = torch.stack([
        assignment == v
        for v in np.unique(assignment)
    ])

    boundaries = masks_to_boundaries(regions)

    overlaid_img = image_from_masks(boundaries, combine_as_binary_mask=True, superimpose_on_image=img)
    show(overlaid_img)

# %%
if __name__ == '__main__':
    # Define the input and output directories
    input_dir = '/shared/rsaas/dino_sam/data/ADE20K/images/training'
    output_dir = '/shared/rsaas/blume5/dino_sam/outputs/slic/ADE20K/train'
    vis_device = 'cpu' # Device for visualization; SLIC runs on the CPU

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
        assignment = torch.tensor(slic.iterate(img), device=vis_device)

        # Save the superpixels to a file
        output_filename = os.path.join(output_dir, os.path.splitext(filename)[0] + '.json')
        with open(output_filename, 'w') as f:
            ret_dict = {
                'assignment': assignment.tolist(),
                'clusters': slic.slic_model.clusters
            }

        # Visualize
        # img = torch.tensor(img.transpose(2, 0, 1), device=vis_device) # HWC -> CHW
        # visualize_superpixels(img, assignment)
# %%
