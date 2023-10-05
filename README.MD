# Files 
- SAM region extraction:  `sam.py`
- Feature extraction from model for each image: `extract_features.py`
- Label regions: `label_regions.py`
- Get features per region: `process_regions.py`
- Train and evaluate binary classifier: `train_classifier.py`
- Get region, per-pixel predictions and evaluate: `per_pixel_pred_eval.py`
- Make region predictions and convert to per-pixel predictions (slow): `per_pixel_prediction.py`
- Get mIOU: `per_pixel_eval.py`
# Installation
- `python>=3.9`
-  `pytorch=1.10 cuda=11.1` 
- `pip install opencv-python pycocotools matplotlib `
# File Location (on vision cluster)
- Dataset: `/shared/rsaas/dino_sam/data`
- Checkpoints: `/shared/rsaas/dino_sam/checkpoints`

## Dense_clip description:
- Run `python -m pip install git+http://github.com/yuqunw/DenseCLIP` before running dense clip feature extraction.