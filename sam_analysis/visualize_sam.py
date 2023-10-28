# %%
# See https://pytorch.org/vision/main/auto_examples/others/plot_repurposing_annotations.html

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import os
from torchvision.io import read_image
from pycocotools import mask as mask_utils
from torchvision.utils import draw_segmentation_masks
import shutil
import utils
import torch
from typing import Literal, List, Union

# plt.rcParams["savefig.bbox"] = 'tight'

def show(
    imgs: Union[torch.Tensor,List[torch.Tensor]],
    title: str = None,
    title_y: float = 1,
    subplot_titles: List[str] = None,
    nrows: int = 1,
):
    if not isinstance(imgs, list):
        imgs = [imgs]

    ncols = len(imgs) // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
    fig.tight_layout()

    for i, img in enumerate(imgs):
        row = i // ncols
        col = i % ncols

        img = img.detach()
        img = F.to_pil_image(img)

        axs[row, col].imshow(np.asarray(img))
        axs[row, col].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        # Set titles for each individual subplot
        if subplot_titles and i < len(subplot_titles):
            axs[row, col].set_title(subplot_titles[i])

    if title:
        fig.suptitle(title, y=title_y)

    return fig, axs

def get_colors(num_colors, cmap_name='rainbow', as_tuples=False):
    '''
    Returns a mapping from index to color (RGB).

    Args:
        num_colors (int): The number of colors to generate

    Returns:
        torch.Tensor: Mapping from index to color of shape (num_colors, 3).
    '''
    cmap = plt.get_cmap(cmap_name)

    colors = np.stack([
        (255 * np.array(cmap(i))).astype(int)[:3]
        for i in np.linspace(0, 1, num_colors)
    ])

    if as_tuples:
        colors = [tuple(c) for c in colors]

    return colors

def image_from_masks(
    masks: torch.Tensor,
    combine_as_binary_mask: bool = False,
    superimpose_on_image: torch.Tensor = None,
    cmap: str = 'rainbow'
):
    '''
    Creates an image from a set of masks.

    Args:
        masks (torch.Tensor): (num_masks, height, width)
        combine_as_binary_mask (bool): Show all segmentations with the same color, showing where any mask is present. Defaults to False.
        superimpose_on_image (torch.Tensor): The image to use as the background, if provided: (C, height, width). Defaults to None.
        cmap (str, optional): Colormap name to use when coloring the masks. Defaults to 'rainbow'.

    Returns:
        torch.Tensor: Image of shape (C, height, width) with the plotted masks.
    '''
    # Masks should be a tensor of shape (num_masks, height, width)
    if combine_as_binary_mask:
        masks = masks.sum(dim=0, keepdim=True).to(torch.bool)

    # If there is only one mask, ensure we get a visible color
    colors = get_colors(masks.shape[0], cmap_name=cmap, as_tuples=True) if masks.shape[0] > 1 else 'aqua'

    if superimpose_on_image is not None:
        alpha = .8
        background = superimpose_on_image
    else:
        alpha = 1
        background = torch.zeros(3, masks.shape[1], masks.shape[2], dtype=torch.uint8)

    masks = draw_segmentation_masks(background, masks, colors=colors, alpha=alpha)

    return masks

def compare_model_outputs(
    img_name: str,
    img_dir: str,
    annots_dir: str,
    sam_output_dirs: List[str],
    model_names: List[str],
    cmap='rainbow', # rainbow has a high lightness; see https://matplotlib.org/stable/users/explain/colors/colormaps.html#lightness-of-matplotlib-colormaps
    combine_as_binary_mask: bool = False, # Combine masks and show as masked vs. not-masked region
    superimpose_on_image: bool = True, # Show masks superimposed on image
    save_path: str = None # Where to save the figure
):
    '''
    Compares the segmentation masks of multiple models to the ground truth annotation.

    If combine_as_binary_mask is True, the masks will be combined into a single binary mask indicating
    masked vs. not-masked regions. Otherwise, each mask will be shown separately.

    If superimpose_on_image is True, the masks will be superimposed on the image. Otherwise, the masks
    will be superimposed on a black background. Note that with the rainbow colormap with a highlightness,
    this allows one to distinguish unmasked regions from masked regions.

    Args:
        img_name (str): File name of the image.
        img_dir (str): Directory with all images.
        annots_dir (str): Directory with segmentation annotations (images).
        sam_output_dirs (List[str]): Directories with SAM output JSONs with RLEs.
        model_names (List[str]): Names of the models to display on the figure.
        cmap (str, optional): The colormap to use. Defaults to 'rainbow'.
        combine_as_binary_mask (bool, optional): Combine masks into a single binary mask. Defaults to False.
        superimpose_on_image (bool, optional): Superimpose masks on the original image instead of a black background. Defaults to True.
        save_path (str, optional): Where to save the figure. Defaults to None.
    '''

    img_path = os.path.join(img_dir, img_name)
    image = read_image(img_path)
    superimpose_on_image = image if superimpose_on_image else None

    json_basename = img_name.replace('.jpg','.json')
    output_imgs = []
    for output_dir in sam_output_dirs:
        masks = utils.open_file(os.path.join(output_dir, json_basename))
        masks = [mask_utils.decode(region['segmentation']) for region in masks]
        masks = torch.stack([torch.tensor(m) for m in masks]).to(torch.bool) # (n_masks, h, w)

        image = image_from_masks(masks, combine_as_binary_mask, superimpose_on_image, cmap)
        output_imgs.append(image)

    # Load annotation and convert to a set of masks
    annot_basename = img_name.replace('.jpg','.png')
    annot = read_image(os.path.join(annots_dir, annot_basename))
    annot_masks = torch.cat([annot == val for val in torch.unique(annot)]) # (n_vals, h, w)

    image = image_from_masks(annot_masks, False, superimpose_on_image, cmap)
    output_imgs.append(image)

    fig, _ = show(output_imgs, subplot_titles=model_names + ['Ground Truth'], nrows=2)

    if save_path:
        fig.savefig(save_path, dpi=500)

        # Copy original image
        new_img_path = save_path.split('.')[0] + '_original.jpg'
        shutil.copy(img_path, new_img_path)

def load_everything(img_name: str, img_dir: str, annot_dir: str, sam_output_dir: str):
    '''
    Shows the original image, the combined segmentation masks superimposed on the image,
    something else, then the ground truth annotation.

    Args:
        img_name: basename of the file
        img_dir: dirname to the image
        annot_dir: dirname to the gt annotation?
        sam_output_dir: dirname to the sam output JSON with RLEs
    '''
    image = read_image(os.path.join(img_dir, img_name))
    show(image, title='Original Image')

    json_basename = img_name.replace('.jpg','.json')
    sam_masks = utils.open_file(os.path.join(sam_output_dir, json_basename))
    all_masks = [mask_utils.decode(region['segmentation']) for region in sam_masks]

    combined_masks = torch.tensor(np.stack(all_masks, axis=0).sum(axis=0).astype(bool)) # Binary mask of all masks combined
    superimposed = draw_segmentation_masks(image, combined_masks, colors='aqua')
    show(superimposed, title='Combined Masks') # Visualize masks superimposed on image
    show(combined_masks.to(torch.uint8), title='Combined Masks (binary)') # Visualize in binary "mask/not-mask" colors without background image

    annot_basename = img_name.replace('.jpg','.png')
    annot = read_image(os.path.join(annot_dir, annot_basename))
    show(annot, title='Annotated Masks')

def load_individual_masks(img_name, sam_output_dir, extra=False):
    '''
    Shows first 20 masks in groups of four. If extra is True, shows the next 16 masks in groups of four.

    Args:
        img_name (str): Base name of the image
        sam_output_dir (str): Directory to the sam output JSONs with RLEs.
        extra (bool, optional): Show the next 16 masks, if any. Defaults to False.
    '''
    file_name = img_name.replace('.jpg','.json')
    sam_masks = utils.open_file(os.path.join(sam_output_dir, file_name))
    sorted_regions = sorted(sam_masks, key=(lambda x: x['area']), reverse=True)
    all_masks = [mask_utils.decode(region['segmentation']) for region in sorted_regions]

    all_masks_separate = []
    for m in all_masks:
        nonzero = np.nonzero(m)
        m_mask = np.zeros_like(all_masks[0])
        m_mask[nonzero[0],nonzero[1]] = 1
        all_masks_separate.append(torch.from_numpy(m_mask.astype('uint8')))

    for i in range(0, min(20, len(all_masks)), 4):
        show(all_masks_separate[i:i+4], title=f'Masks {i}-{i+3}')

    if extra:
        for i in range(20, min(36, len(all_masks)), 4):
            show(all_masks_separate[i:i+4], title=f'Masks {i}-{i+3}')

# %% ADE20K
if __name__ == '__main__':
    ex_ind = 0 # Example index
    model: Literal['sam', 'mobile_sam', 'hq_sam'] = 'mobile_sam'
    img_dir = '/shared/rsaas/dino_sam/data/ADE20K/images/training'
    annots_dir = '/shared/rsaas/dino_sam/data/ADE20K/annotations/training'
    model_output_dir = f'/shared/rsaas/dino_sam/{model}_output/ADE20K/train'

    img_names = os.listdir(img_dir)

    # %%
    load_individual_masks(img_names[ex_ind],  model_output_dir)
    load_everything(img_names[ex_ind], img_dir, annots_dir, model_output_dir)

    # %% Compare SAM models
    model_names = ['sam', 'mobile_sam', 'hq_sam']
    for ex_ind in range(10):
        compare_model_outputs(
            img_name=img_names[ex_ind],
            img_dir=img_dir,
            annots_dir=annots_dir,
            sam_output_dirs=[f'/shared/rsaas/dino_sam/{m}_output/ADE20K/train' for m in model_names],
            model_names=['SAM', 'MobileSAM', 'HQ-SAM'],
            combine_as_binary_mask=False,
            superimpose_on_image=False,
            save_path=f'ade20k_{ex_ind}.png'
        )

    # %% Pascal VOC
    ex_ind = 0
    model: Literal['sam', 'mobile_sam', 'hq_sam'] = 'sam'
    file_prefix = '/shared/rsaas/dino_sam'
    images_prefix = 'data/VOCdevkit/VOC2012'

    img_dir = os.path.join(file_prefix, images_prefix,'segmentation_imgs/train')
    annots_dir = os.path.join(file_prefix, images_prefix, 'segmentation_annotation/train')
    model_output_dir = os.path.join(file_prefix, 'sam_output/pascal_voc/train')

    img_names = os.listdir(img_dir)

    # %%
    load_individual_masks(img_names[ex_ind], model_output_dir)
    load_everything(img_names[ex_ind], img_dir, annots_dir, model_output_dir)

    # %% Compare SAM models
    model_names = ['sam', 'mobile_sam', 'hq_sam']
    for ex_ind in range(10):
        compare_model_outputs(
            img_name=img_names[ex_ind],
            img_dir=img_dir,
            annots_dir=annots_dir,
            sam_output_dirs=[f'/shared/rsaas/dino_sam/{m}_output/pascal_voc/train' for m in model_names],
            model_names=['SAM', 'MobileSAM', 'HQ-SAM'],
            combine_as_binary_mask=False,
            superimpose_on_image=False,
            save_path=f'pascal_{ex_ind}.png'
        )