# Files 
- SAM region extraction:  `sam.py`
- Feature extraction from model for each image: `extract_features.py`
- Label regions: `label_regions.py`
- Get features per region: `process_regions.py`
- Train and evaluate binary classifier: `train_classifier.py`
- Make region predictions and convert to per-pixel predictions: `per_pixel_prediction.py`
- Get mIOU: `per_pixel_eval.py`
# Installation
- `python>=3.9`
-  `pytorch=1.10 cuda=11.1` 
- `pip install opencv-python pycocotools matplotlib `
# File Location (on vision cluster)
- Dataset: `/shared/rsaas/dino_sam/data`
- Checkpoints: `/shared/rsaas/dino_sam/checkpoints`

