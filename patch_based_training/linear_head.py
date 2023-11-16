'''
Based on https://github.com/facebookresearch/dinov2/blob/da4b3825f0ed64b7398ace00c5062503811d0cff/dinov2/eval/segmentation/models/decode_heads/linear_head.py#L15.

Returns:
    _type_: _description_
'''
import torch
import torch.nn.functional as F
from typing import List
from einops import rearrange

class LinearHead(torch.nn.Module):
    '''
    Batchnorm followed by fully connected layer.
    Based on https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/segmentation/models/decode_heads/linear_head.py
    and https://github.com/open-mmlab/mmsegmentation/blob/cbf9af1e3f2eb50b58c8d783008f749ef62f2435/mmseg/models/decode_heads/decode_head.py#L18C7-L18C7

    The BaseDecodeHead of MMSeg uses a 1x1 kernel, which is just a fully connected layer.
    '''
    def __init__(self, n_classes: int, use_batchnorm: bool = False):
        super().__init__()

        self.possible_bn = torch.nn.LazyBatchNorm2d() if use_batchnorm else torch.nn.Identity()
        self.fc = torch.nn.LazyLinear(n_classes)

    @torch.no_grad()
    def _transform_inputs(self, img_feats: List[torch.Tensor]):
        """
        Outputs a single tensor of shape (N, C, H, W) from a list of tensors of shape (N, C_i, H_i, W_i).
        Resized to the shape of the first tensor in the list.

        Args:
            img_feats (List[Tensor]): List of img features of different sizes.
        Returns:
            Tensor: The transformed inputs
        """

        img_feats = torch.cat([
            F.interpolate(x, size=img_feats[0].shape[-2:], mode="bilinear")
            for x in img_feats
        ], dim=0) if len(img_feats) > 1 else img_feats[0]

        return img_feats

    def forward(self, img_feats: List[torch.Tensor]):
        '''
        Args:
            img_feats: List of img features of different sizes.
        '''
        x = self._transform_inputs(img_feats) # (n, c, h, w)
        x = self.possible_bn(x) # (n, c, h, w)
        x = rearrange(x, 'n c h w -> n h w c')
        x = self.fc(x) # (n, h, w, n_classes)

        return x