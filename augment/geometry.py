import numpy as np
import math
from copy import deepcopy
from PIL import Image
from typing import *

def get_im_dims_from_bundle(
        image_bundle: dict[str, Image.Image]
    ):
    im_width, im_height = image_bundle[next(iter(image_bundle))].size
    return im_width, im_height

def pick_bucket_dims(
    im_width: int, 
    im_height: int, 
    bucket_dim_options: List[Tuple[int, int]], 
    prefer_similar_aspect: bool
):
    assert len(bucket_dim_options) > 0
    if prefer_similar_aspect:
        """Find which allowed bucket dimensions are most similar to the image's aspect ratio."""
        im_aspect = im_width / im_height
        bucket_aspects = np.array([w / h for w, h in bucket_dim_options])
        closest_aspect = bucket_aspects[np.argmin(np.abs(bucket_aspects - im_aspect))]
        idxs_of_closest = np.array(list(filter(lambda idx: bucket_aspects[idx] == closest_aspect, np.array(range(len(bucket_aspects)))))).astype(int)
        bucket_dim_options = np.array(bucket_dim_options)[idxs_of_closest]

    bucket_dims = bucket_dim_options[np.random.randint(len(bucket_dim_options))]
    return tuple(bucket_dims)

def random_downscale(
        image_bundle: dict[str, Image.Image],
        bucket_dims: tuple[int, int],
        max_downscale_factor: Optional[float] = None,
        distribution: str = 'normal'
    ):
        im_width, im_height = get_im_dims_from_bundle(image_bundle)
        bucket_width, bucket_height = bucket_dims
        if max_downscale_factor is not None:
            max_downscale_factor = max(min(im_width/bucket_width, im_height/bucket_height), max_downscale_factor)
        else:
            max_downscale_factor = min(im_width/bucket_width, im_height/bucket_height)
        print('max_downscale_factor', max_downscale_factor)
        if distribution == 'normal':
            downscale_factor = np.random.normal(scale=max_downscale_factor / 2.0)
            downscale_factor = np.abs(downscale_factor) + 1.0
            downscale_factor = min(downscale_factor, max_downscale_factor)
        elif distribution == 'uniform':
            downscale_factor = np.random.uniform(1.0, max_downscale_factor)
        else:
            raise ValueError
        ds_width = int(im_width / downscale_factor)
        ds_height = int(im_height / downscale_factor)
        for k in image_bundle:
            image_bundle[k] = image_bundle[k].resize((ds_width, ds_height), Image.LANCZOS)
        print('Resized image to  ',(ds_width, ds_height))

        return downscale_factor

def random_rotate_crop_flip(
    image_bundle: dict[str, Image.Image],
    bucket_dims: tuple[int, int],
    do_random_90_degree_rotations: bool = False,
    do_random_flip_lr: bool = False
):
    bucket_width, bucket_height = bucket_dims
    im_width, im_height = get_im_dims_from_bundle(image_bundle)
    margin_lr = (im_width - bucket_width) // 2
    margin_ud = (im_height - bucket_height) // 2
    print('margins:', margin_lr, margin_ud)
    if margin_lr > 0:
        permute_center_lr = np.random.randint(-margin_lr, margin_lr)
    else:
        permute_center_lr = 0.0
    if margin_ud > 0:
        permute_center_ud = np.random.randint(-margin_ud, margin_ud)
    else:
        permute_center_ud = 0.0
    rot_margin_lr = margin_lr - abs(permute_center_lr)
    rot_margin_ud = margin_ud - abs(permute_center_ud)

    left = im_width // 2 - permute_center_lr - bucket_width // 2
    top = im_height // 2 - permute_center_ud - bucket_height // 2
    right = left + bucket_width
    bottom = top + bucket_height
    """
    Math note:
    When you rotate a rectangle W x H, the bounding box takes the dimensions W' = W |cos Θ| + H |sin Θ|, H' = W |sin Θ| + H |cos Θ|.

    If you need to fit that in a W" x H" rectangle, the scaling factor is the smallest of W"/W' and H"/H'.

    Let's say W = H = D. 
    Then the bounding box takes dimensions D' = D|cos Θ| + D|sin Θ|
        D' = D(|cos Θ| + |sin Θ|)
        D'/D = |cos Θ| + |sin Θ|
    Solve for Θ:
    Θ = (1/2) sin^-1((D'/D)^2 - 1)
    """
    print('rot margins:', rot_margin_lr, rot_margin_ud)
    min_margin_lr = min(left, im_width - right)
    min_margin_ud = min(top, im_height - bottom)

    rot_scale_ratio = min((bucket_width + min_margin_lr) / bucket_width, (bucket_height + min_margin_ud) / bucket_height)
    print('rot scale ratio', rot_scale_ratio)

    max_rot_angle = math.degrees((1/2) * math.asin(min(1,max((rot_scale_ratio*rot_scale_ratio) - 1,-1))))
    rot_angle = np.random.uniform(-max_rot_angle, max_rot_angle)
    print('max rot angle:', max_rot_angle)

    if do_random_90_degree_rotations:
        transpose_method = [None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270][np.random.randint(4)]
    else:
        transpose_method = None

    do_flip_lr = do_random_flip_lr and np.random.uniform() > 0.5

    for k in image_bundle:
        image_bundle[k] = image_bundle[k].rotate(rot_angle, resample=Image.BICUBIC, expand=True)
        image_bundle[k] = image_bundle[k].crop((left, top, right, bottom))
        if do_flip_lr:
            image_bundle[k] = image_bundle[k].transpose(method = Image.Transpose.FLIP_LEFT_RIGHT)
        if transpose_method is not None:
            image_bundle[k] = image_bundle[k].transpose(method = transpose_method)

    return ((left, top, right, bottom), rot_angle, do_flip_lr, transpose_method)


def geo_transforms(
        image_bundle: dict[str, Image.Image],
        bucket_dim_options: List[Tuple[int, int]],
        highest_saliency_bucket_probes: int = 1,
        max_downscale_factor: Optional[float] = None,
        prefer_similar_aspect: bool = False,
        do_random_90_degree_rotations: bool = False,
        do_random_flip_lr: bool = False
    ):
    """
        image_bundle: 
            dictionary of parallel images representing a training sample. The same randomized geomtric transform will be applied to all images in the sample. All images in sample should have same dimensions. Dictionary keys label the role of the image in the training sample. "saliency_map" is a special key for the saliency map.
        bucket_dim_options: 
            allowed spatial dimensions for the bucket. Example: [(512, 512), (512, 768), (658, 658)] etc. Dimensions whose width or height exceeds the width or height of the training sample will be filtered out.
        highest_saliency_bucket_probes:
    """
    im_width, im_height = get_im_dims_from_bundle(image_bundle)
    # Check that all spatial dimensions of all images in image_bundle dict are equal. 
    for _, im in image_bundle.items():
        print(im.size, im_width, im_height)
        assert im.size == (im_width, im_height,)

    transformed_images = deepcopy(image_bundle)

    # Filter out bucket dim options that are larger than the image size
    bucket_dim_options = list(filter(lambda t: t[0] <= im.width and t[1] <= im.height, bucket_dim_options))

    # Choose bucket dims for this bucket extraction from the sample.
    bucket_width, bucket_height = pick_bucket_dims(im_width, im_height, bucket_dim_options, prefer_similar_aspect)

    downscale_factor = random_downscale(transformed_images, (bucket_width, bucket_height), max_downscale_factor)

    (left, top, right, bottom), rot_angle, did_flip_lr, transpose_method = random_rotate_crop_flip(transformed_images, (bucket_width, bucket_height), do_random_90_degree_rotations, do_random_flip_lr)

    return transformed_images, {'downscale_factor': downscale_factor, 'bucket_dims': (bucket_width, bucket_height), ' crop_box': (left, top, right, bottom), 'rot_angle': rot_angle,'transpose_method': transpose_method, 'did_flip_lr': did_flip_lr}



