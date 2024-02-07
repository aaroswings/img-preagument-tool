import numpy as np
import math

from typing import *

def pick_bucket_dims(
    im_width: int, 
    im_height: int, 
    bucket_dim_options: List[Tuple[int, int]], 
    prefer_similar_aspect: bool = False
):
    if prefer_similar_aspect:
        """Find which allowed bucket dimensions are most similar to the image's aspect ratio."""
        im_aspect = im_width / im_height
        bucket_aspects = np.array([w / h for w, h in bucket_dim_options])
        closest_aspect = bucket_aspects[np.argmin(np.abs(bucket_aspects - im_aspect))]
        idxs_of_closest = np.array(list(filter(lambda idx: bucket_aspects[idx] == closest_aspect, np.array(range(len(bucket_aspects)))))).astype(int)
        print(idxs_of_closest)
        bucket_dim_options = np.array(bucket_dim_options)[idxs_of_closest]

    bucket_dims = bucket_dim_options[np.random.randint(len(bucket_dim_options))]
    return tuple(bucket_dims)

