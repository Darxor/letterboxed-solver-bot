import numpy as np
from skimage import color


def resize_box(box: tuple | list, scale: float) -> tuple[float, float, float, float]:
    x, y, w, h = box
    center_x = x + w / 2
    center_y = y + h / 2

    new_w = w * scale
    new_h = h * scale

    new_x = center_x - new_w / 2
    new_y = center_y - new_h / 2

    return new_x, new_y, new_w, new_h


def img2gray(image: np.ndarray) -> np.ndarray:
    match image.shape[2]:
        case 1:
            image = image
        case 3:
            image = color.rgb2gray(image)
        case 4:
            image = color.rgb2gray(color.rgba2rgb(image))

    return image

def array2image(image: np.ndarray) -> np.ndarray:
    return (image * 255).astype(np.uint8)