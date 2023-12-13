import pickle
import os
from json import JSONEncoder
from typing import Dict, Optional
import yaml
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import math
import torch
from typing import List, Dict, Any
import logging
import coloredlogs
from PIL import Image, ImageDraw
import warnings
import stat

logger = logging.getLogger(__name__)
coloredlogs.install(logging.INFO, logger=logger)

def save_file(filename,data,json_numpy=False):
    """
    Based on https://github.com/salesforce/LAVIS/blob/main/lavis/common/utils.py
    Supported:
        .pkl, .pickle, .npy, .json
    """
    parent_dir = os.path.dirname(filename)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    # Path(parent_dir).chmod(0o0777)
    file_ext = os.path.splitext(filename)[1]
    if file_ext == ".npy":
        with open(filename, "wb+") as fopen:
            np.save(fopen, data)
    elif file_ext == ".json":
        if not json_numpy:
            with open(filename,'w+') as fopen:
                json.dump(data,fopen,indent=2)
        else:
            with open(filename, "w+") as fopen:
                json.dump(data,fopen,cls=NumpyArrayEncoder)
    elif file_ext == ".yaml":
        with open(filename, "w+") as fopen:
            dump = yaml.dump(data)
            fopen.write(dump)
            fopen.flush()
    else:
        # assume file is pickle
         with open(filename, "wb+") as fopen:
            pickle.dump(data, fopen)
    # give everybody read,write,execute
    # not secure but should be ok
    # Path(filename).chmod(0o0777)



def open_file(filename):
    """
    Based on https://github.com/salesforce/LAVIS/blob/main/lavis/common/utils.py
    Supported:
        .pkl, .pickle, .npy, .json
    """
    file_ext = os.path.splitext(filename)[1]
    if file_ext == '.txt':
        with open(filename,'r+') as fopen:
            data = fopen.readlines()
    elif file_ext in [".npy",".npz"]:
        data = np.load(filename,allow_pickle=True)
    elif file_ext == '.json':
        with open(filename,'r+') as fopen:
            data = json.load(fopen)
    elif file_ext == ".yaml":
        with open(filename,'r+') as fopen:
            data = yaml.load(fopen,Loader=yaml.FullLoader)
    else:
        # assume pickle
        with open(filename,"rb+") as fopen:
            data = pickle.load(fopen)
    return data

def save_pkl_file(filename,file_contents):
    parent_dir = filename.split('/')[-2]
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    with open(filename, 'wb') as f:
            pickle.dump(file_contents, f)

def open_pkl_file(filename):
    with open(filename,'rb') as r:
        file_contents = pickle.load(r)
    return file_contents
def open_json_file(filename):
    with open(filename,'r+') as r:
        file_contents = json.load(r)
    return file_contents

def save_json_file(filename,file_contents,numpy=False):
    parent_dir = filename.split('/')[-2]
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    if not numpy:
        with open(filename,'w+') as w:
            json.dump(file_contents,w)
    else:
          with open(filename,'w+') as w:
            json.dump(file_contents,w,cls=NumpyArrayEncoder)

def polygon_to_mask(polygons,h,w):
    # for vaw, takes size of image and plots polygon boundary
    # from https://github.com/ChenyunWu/PhraseCutDataset/blob/master/utils/data_transfer.py#L48
    p_mask = np.zeros((h, w))
    for polygon in polygons:
        if len(polygon) < 2:
            continue
        p = []
        for x, y in polygon:
            p.append((int(x), int(y)))
        img = Image.new('L', (w, h), 0)
        ImageDraw.Draw(img).polygon(p, outline=1, fill=1)
        mask = np.array(img)
        p_mask += mask
    p_mask = p_mask > 0
    return p_mask.astype(int)



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
def intersect_and_union(
    pred_label,
    label,
    num_labels,
    ignore_index: bool,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
    reduce_pred_labels: bool = False,
):
    """Calculate intersection and Union.
    Args:
        pred_label (`ndarray`):
            Prediction segmentation map of shape (height, width).
        label (`ndarray`):
            Ground truth segmentation map of shape (height, width).
        num_labels (`int`):
            Number of categories.
        ignore_index (`int`):
            Index that will be ignored during evaluation.
        label_map (`dict`, *optional*):
            Mapping old labels to new labels. The parameter will work only when label is str.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
            and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
        reduce_pred_labels (`bool`, *optional*, defaults to `False`):
            Do the same as `reduce_labels` but for prediction labels.
     Returns:
         area_intersect (`ndarray`):
            The intersection of prediction and ground truth histogram on all classes.
         area_union (`ndarray`):
            The union of prediction and ground truth histogram on all classes.
         area_pred_label (`ndarray`):
            The prediction histogram on all classes.
         area_label (`ndarray`):
            The ground truth histogram on all classes.
    """
    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id

    # turn into Numpy arrays
    pred_label = np.array(pred_label)
    label = np.array(label)

    if reduce_labels:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    if reduce_pred_labels:
        pred_label[pred_label == 0] = 255
        pred_label = pred_label - 1
        pred_label[pred_label == 254] = 255

    mask = label != ignore_index
    #mask = np.not_equal(label, ignore_index)
    pred_label = pred_label[label!=ignore_index]

    label = label[label!= ignore_index]

    #label = np.array(label)[mask]

    intersect = pred_label[pred_label == label]

    area_intersect = np.histogram(intersect, bins=num_labels, range=(0, num_labels - 1))[0]
    area_pred_label = np.histogram(pred_label, bins=num_labels, range=(0, num_labels - 1))[0]
    area_label = np.histogram(label, bins=num_labels, range=(0, num_labels - 1))[0]

    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(
    results,
    gt_seg_maps,
    num_labels,
    ignore_index: bool,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
    reduce_pred_labels: bool = False
):
    """Calculate Total Intersection and Union, by calculating `intersect_and_union` for each (predicted, ground truth) pair.
    Args:
        results (`ndarray`):
            List of prediction segmentation maps, each of shape (height, width).
        gt_seg_maps (`ndarray`):
            List of ground truth segmentation maps, each of shape (height, width).
        num_labels (`int`):
            Number of categories.
        ignore_index (`int`):
            Index that will be ignored during evaluation.
        label_map (`dict`, *optional*):
            Mapping old labels to new labels. The parameter will work only when label is str.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
            and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
        reduce_pred_labels (`bool`, *optional*, defaults to `False`):
            Same as `reduce_labels` but for prediction labels.
     Returns:
         total_area_intersect (`ndarray`):
            The intersection of prediction and ground truth histogram on all classes.
         total_area_union (`ndarray`):
            The union of prediction and ground truth histogram on all classes.
         total_area_pred_label (`ndarray`):
            The prediction histogram on all classes.
         total_area_label (`ndarray`):
            The ground truth histogram on all classes.
    """
    total_area_intersect = np.zeros((num_labels,), dtype=np.float64)
    total_area_union = np.zeros((num_labels,), dtype=np.float64)
    total_area_pred_label = np.zeros((num_labels,), dtype=np.float64)
    total_area_label = np.zeros((num_labels,), dtype=np.float64)
    for result, gt_seg_map in tqdm(zip(results, gt_seg_maps), total=len(results)):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            result, gt_seg_map, num_labels, ignore_index, label_map, reduce_labels, reduce_pred_labels
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, total_area_label


def mean_iou(
    results,
    gt_seg_maps,
    num_labels,
    ignore_index: bool,
    nan_to_num: Optional[int] = None,
    label_map: Optional[Dict[int, int]] = None,
    reduce_labels: bool = False,
    reduce_pred_labels: bool = False,
):
    """Calculate Mean Intersection and Union (mIoU).
    Args:
        results (`ndarray`):
            List of prediction segmentation maps, each of shape (height, width).
        gt_seg_maps (`ndarray`):
            List of ground truth segmentation maps, each of shape (height, width).
        num_labels (`int`):
            Number of categories.
        ignore_index (`int`):
            Index that will be ignored during evaluation.
        nan_to_num (`int`, *optional*):
            If specified, NaN values will be replaced by the number defined by the user.
        label_map (`dict`, *optional*):
            Mapping old labels to new labels. The parameter will work only when label is str.
        reduce_labels (`bool`, *optional*, defaults to `False`):
            Whether or not to reduce all label values of segmentation maps by 1. Usually used for datasets where 0 is used for background,
            and background itself is not included in all classes of a dataset (e.g. ADE20k). The background label will be replaced by 255.
        reduce_pred_labels (`bool`, *optional*, defaults to `False`):
            Same as `reduce_labels` but for prediction labels.
    Returns:
        `Dict[str, float | ndarray]` comprising various elements:
        - *mean_iou* (`float`):
            Mean Intersection-over-Union (IoU averaged over all categories).
        - *mean_accuracy* (`float`):
            Mean accuracy (averaged over all categories).
        - *overall_accuracy* (`float`):
            Overall accuracy on all images.
        - *per_category_accuracy* (`ndarray` of shape `(num_labels,)`):
            Per category accuracy.
        - *per_category_iou* (`ndarray` of shape `(num_labels,)`):
            Per category IoU.
    """
    total_area_intersect, total_area_union, total_area_pred_label, total_area_label = total_intersect_and_union(
        results, gt_seg_maps, num_labels, ignore_index, label_map, reduce_labels, reduce_pred_labels
    )

    # compute metrics
    metrics = dict()

    all_acc = total_area_intersect.sum() / total_area_label.sum()
    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_label

    metrics["mean_iou"] = np.nanmean(iou)
    metrics["mean_accuracy"] = np.nanmean(acc)
    metrics["overall_accuracy"] = all_acc
    metrics["per_category_iou"] = iou
    metrics["per_category_accuracy"] = acc

    if nan_to_num is not None:
        metrics = dict(
            {metric: np.nan_to_num(metric_value, nan=nan_to_num) for metric, metric_value in metrics.items()}
        )

    return metrics

def split_gt_masks(gt_img:np.array, ignore_zero_label:bool):
    """
    Copy of function from sam_analysis/max_possible_iou.py. Splits instance mask into binary masks
    Used to process ADE20K instance annotations
    """
    labels = np.asarray([l for l in np.unique(gt_img) if l!= 0 or not ignore_zero_label])
    if any(labels < 0):
        raise ValueError('Label value less than zero detected. This is not supported.')

    if len(labels) == 0:
        err_str = 'Failed to split GT image into masks; no labels detected.'
        logger.error(err_str)
    if not ignore_zero_label:
        labels += 1
        gt_img += 1
    gt_masks = np.stack([(gt_img == l) * l for l in labels]) # (nlabels, h, w)

    return gt_masks, labels





def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def intersect_2d(x1, x2):
    """
    Given two arrays [m1, n], [m2,n], returns a [m1, m2] array where each entry is True if those
    rows match.
    :param x1: [m1, n] numpy array
    :param x2: [m2, n] numpy array
    :return: [m1, m2] bool array of the intersections
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input arrays must have same #columns")

    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    res = (x1[..., None] == x2.T[None, ...]).all(1)
    return res