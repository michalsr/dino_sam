# %%
import os
import numpy as np
from PIL import Image
from fast_slic import Slic
from tqdm import tqdm

# Define the input and output directories
input_dir = '/shared/rsaas/dino_sam/data/ADE20K/images/training'
output_dir = '/shared/rsaas/blume5/dino_sam/outputs/slic/ADE20K/train'

# Define the parameters for the superpixel algorithm
num_components = 1000
compactness = 10

os.makedirs(output_dir, exist_ok=True)

# Loop through each image in the input directory
for filename in tqdm(os.listdir(input_dir)):
    # Load the image
    with Image.open(os.path.join(input_dir, filename)) as f:
        img = np.array(f)

    # Apply the superpixel algorithm
    slic = Slic(num_components=num_components, compactness=compactness)
    assignment = slic.iterate(img)

    # Save the superpixels to a file
    output_filename = os.path.join(output_dir, os.path.splitext(filename)[0] + '.json')
    with open(output_filename, 'w') as f:
        ret_dict = {
            'assignment': assignment.tolist(),
            'clusters': slic.slic_model.clusters
        }

# %%
