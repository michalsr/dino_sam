import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main_dir",
        type=str,
        default="/shared/rsaas/dino_sam"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Location of jpg files",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Name for storing features, sam regions, etc."
    )
    parser.add_argument(
        "--annotation_location",
        type=str,
        default=None,
        help="Location of per-pixel files if dense recognition or image labels",
    )
    parser.add_argument(
        "--model_repo_name",
        type=str,
        default="facebookresearch/dinov2",
        help="PyTorch model name for downloading from PyTorch hub"
    )
    parser.add_argument(
        "--model",
        type=str,
        default='dinov2_vitl14',
        help="Name of model from repo"
    )
    parser.add_argument(
        "--intermediate_layers",
        action="store_false",
        help="Use CLS token or earlier features"
    )
    parser.add_argument(
        "--last_n_layers",
        type=str,
        default="4",
        help="How many layers to use if intermediate_layers=True. Can also pass in list of layers"
    )
    parser.add_argument(
        "--classifier_dir",
        type=str,
        default="/shared/rsaas/dino_sam/classifiers"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="/shared/rsaas/dino_sam/features"
    )
    parser.add_argument(
        "--pooled_dir",
        type=str,
        default="/shared/rsaas/dino_sam/pooled_features_pkl"
    )
    parser.add_argument(
        "--sam_location",
        type=str,
        default="/shared/rsaas/dino_sam/sam_output"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=0,
        help="Total number of classes"
    )
    parser.add_argument(
        "--label_dir",
        default='/shared/rsaas/dino_sam/region_labels'
    )
    parser.add_argument(
        "--ignore_zero",
        action="store_false",
        help="If 0 is background we can ignore it"
    )
    parser.add_argument(
        "--ignore_index",
        nargs="+",
        action="append",
        required=False,
        help="Add for each ignore index like 255"
    )
    parser.add_argument(
        "--save_results",
        default="",
        help="If empty then will not save results"
    )
    parser.add_argument(
        "--label_percent",
        default=95,
        help='Percetange of pixels in region that need to be of a certain class before region is that class'
    )
    parser.add_argument(
        "--padding",
        default="center",
        help="Padding used for transforms"
    )
    parser.add_argument(
        "--pixel_to_class",
        default="/shared/rsaas/dino_sam/feature_preds"
    )



    #sam regions 
    parser.add_argument(
    "--input",
    type=str,
    default=None
    help="Path to either a single input image or folder of images.",
)

    parser.add_argument(
    "--output",
    type=str,
    default=None,
    help=(
        "Path to the directory where masks will be output. Output will be either a folder "
        "of PNGs per image or a single json with COCO-style masks."
    ))

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )

    parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

    parser.add_argument(
        "--convert-to-rle",
        action="store_true",
        help=(
            "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
            "Requires pycocotools."
        ),
    )

    amg_settings = parser.add_argument_group("AMG Settings")

    amg_settings.add_argument(
        "--points-per-side",
        type=int,
        default=None,
        help="Generate masks by sampling a grid over the image with this many points to a side.",
    )

    amg_settings.add_argument(
        "--points-per-batch",
        type=int,
        default=None,
        help="How many input points to process simultaneously in one batch.",
    )

    amg_settings.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=None,
        help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-thresh",
        type=float,
        default=None,
        help="Exclude masks with a stability score lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-offset",
        type=float,
        default=None,
        help="Larger values perturb the mask more when measuring stability score.",
    )

    amg_settings.add_argument(
        "--box-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding a duplicate mask.",
    )

    amg_settings.add_argument(
        "--crop-n-layers",
        type=int,
        default=None,
        help=(
            "If >0, mask generation is run on smaller crops of the image to generate more masks. "
            "The value sets how many different scales to crop at."
        ),
    )

    amg_settings.add_argument(
        "--crop-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding duplicate masks across different crops.",
    )

    amg_settings.add_argument(
        "--crop-overlap-ratio",
        type=int,
        default=None,
        help="Larger numbers mean image crops will overlap more.",
    )

    amg_settings.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=None,
        help="The number of points-per-side in each layer of crop is reduced by this factor.",
    )

    amg_settings.add_argument(
        "--min-mask-region-area",
        type=int,
        default=None,
        help=(
            "Disconnected mask regions or holes with area smaller than this value "
            "in pixels are removed by postprocessing."
        ),
    )
    parsed_args = parser.parse_args()
    return parsed_args

