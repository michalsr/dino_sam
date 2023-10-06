import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'
import torchvision.transforms.functional as F
import os 
from torchvision.io import read_image
from PIL import Image
from pycocotools import mask as mask_utils
from torchvision.utils import draw_segmentation_masks
import utils 

"""
Functions to use with jupyter notebook for visualizing 
"""

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if torch.is_tensor(img):
            img = img.detach()
            img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

def load_image(image_name,img_path):
    image =read_image(os.path.join(img_path,image_name))
    show(image)

def draw_all_masks_on_image(image_name,sam_path,image_path,color='aqua'):
    file_name = image_name.replace('.jpg','.json')
    sam_masks = utils.open_file(os.path.join(sam_path,file_name))
    sorted_anns = sorted(sam_masks, key=(lambda x: x['area']), reverse=True)
    all_masks = [mask_utils.decode(region['segmentation']) for region in sam_masks]
    all_masks_stack = np.stack(all_masks,axis=-1).sum(axis=-1)
    nonzero = np.nonzero(all_masks_stack)
    
    combine_masks = np.zeros_like(all_masks_stack)
    combine_masks[nonzero[0],nonzero[1]] = 1 
    show(draw_segmentation_masks(image_1,torch.from_numpy(combine_masks.astype(bool)),colors=color))

def load_masks(image_name,sam_path):
    file_name = image_name.replace('.jpg','.json')
    sam_masks = utils.open_file(os.path.join(sam_path,file_name))
    sorted_anns = sorted(sam_masks, key=(lambda x: x['area']), reverse=True)
    all_masks = [mask_utils.decode(region['segmentation']) for region in sam_masks]
    all_masks_stack = np.stack(all_masks,axis=-1).sum(axis=-1)
    nonzero = np.nonzero(all_masks_stack)
    
    combine_masks = np.zeros_like(all_masks_stack)
    combine_masks[nonzero[0],nonzero[1]] = 1 
    show(torch.from_numpy(combine_masks.astype('uint8')))

def load_individual_masks(image_name,sam_path,extra=False):
    file_name = image_name.replace('.jpg','.json')
    sam_masks = utils.open_file(os.path.join(sam_path,file_name))
    sorted_anns = sorted(sam_masks, key=(lambda x: x['area']), reverse=True)
    all_masks = [mask_utils.decode(region['segmentation']) for region in sorted_anns]
    all_masks_separate = []
    for m in all_masks:
        nonzero = np.nonzero(m)
        m_mask = np.zeros_like(all_masks[0])
        m_mask[nonzero[0],nonzero[1]] = 1
        all_masks_separate.append(torch.from_numpy(m_mask.astype('uint8')))
    show(all_masks_separate[:4])
    show(all_masks_separate[4:8])
    show(all_masks_separate[8:12])
    show(all_masks_separate[12:16])
    show(all_masks_separate[16:20])
    if extra:
        show(all_masks_separate[20:24])
        show(all_masks_separate[24:28])
        show(all_masks_separate[28:32])
        show(all_masks_separate[32:36])


