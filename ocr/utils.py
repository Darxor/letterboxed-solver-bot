import io
import logging
from datetime import datetime
from functools import wraps

import numpy as np
from skimage import io as skio


def rescale_box(
    box: tuple[int, int, int, int], factor: float = 1.0
) -> tuple[int, int, int, int]:
    minr, minc, maxr, maxc = box

    # Calculate center of the box
    center_row = (minr + maxr) / 2
    center_col = (minc + maxc) / 2

    # Calculate current height and width
    height = maxr - minr
    width = maxc - minc

    # Calculate new height and width
    new_height = height * factor
    new_width = width * factor

    # Calculate new min and max rows and columns
    new_minr = int(center_row - new_height / 2)
    new_maxr = int(center_row + new_height / 2)
    new_minc = int(center_col - new_width / 2)
    new_maxc = int(center_col + new_width / 2)

    return new_minr, new_minc, new_maxr, new_maxc


def bounding_square(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """
    Convert bounding box to square
    """
    minr, minc, maxr, maxc = box
    width = maxc - minc
    height = maxr - minr

    # calculate new bounding square box
    if width > height:
        diff = width - height
        minr -= diff // 2
        maxr += diff // 2
    else:
        diff = height - width
        minc -= diff // 2
        maxc += diff // 2

    return minr, minc, maxr, maxc


def array2image(image: np.ndarray) -> np.ndarray:
    return (image * 255).astype(np.uint8)

def imgarray2bytesio(image: np.ndarray) -> io.BytesIO:
    image = array2image(image)
    image_bytes = skio.imsave("<bytes>", image, plugin="imageio", extension=".png")
    return io.BytesIO(image_bytes)

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        logging.info(f"{func.__name__} ran in: {end - start}")
        return result
    return wrapper