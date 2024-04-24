import io

import numpy as np
import pytesseract
from skimage import feature, filters, measure, morphology, segmentation, transform
from skimage import io as skio

from .utils import array2image, resize_box

TEMPLATE_PATH = "templates/template3.png"


async def process_image(image: io.BytesIO):
    img = skio.imread(image, as_gray=True, plugin="imageio")
    template = skio.imread(TEMPLATE_PATH, as_gray=True, plugin="imageio")

    template_loc, template_dim = find_template(img, template)
    img_cropped = crop_image(img, template_loc, template_dim, 1.5)
    img_cleaned = clean_image(img_cropped)
    img_cleaned = array2image(img_cleaned)

    text = pytesseract.image_to_string(
        img_cleaned,
        lang="eng",
        config="-c tessedit_char_whitelist=lckmopsuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01 --psm 6",
    )
    text_ocr = (
        text.strip().replace("0", "O").replace("1", "I").replace("l", "I").upper()
    )

    outimage = skio.imsave("<bytes>", img_cleaned, plugin="imageio", extension=".png")
    return io.BytesIO(outimage), text_ocr


def crop_image(
    image: np.ndarray, loc: np.ndarray, dim: np.ndarray, dim_resize: float = 1.0
) -> np.ndarray:
    dim_resize = dim_resize - 1
    crop = [dim[0] * dim_resize, dim[1] * dim_resize]
    x, y = loc

    x_min = int(x - crop[1] // 2)
    x_max = int(x + dim[1] + crop[1] // 2)
    y_min = int(y - crop[0] // 2)
    y_max = int(y + dim[0] + crop[0] // 2)

    return image[y_min:y_max, x_min:x_max]


def find_template(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    scales = np.linspace(0.25, 2, 20)

    max_corr = 0
    best_location = (0, 0)
    best_template_dim = (template.shape[1], template.shape[0])

    # Perform template matching at different scales
    for scale in scales:
        template_rescaled = transform.rescale(template, scale)

        if (
            template_rescaled.shape[0] > image.shape[0]
            or template_rescaled.shape[1] > image.shape[1]
        ):
            continue

        result = feature.match_template(
            image,
            template_rescaled,
        )

        # Check for the highest correlation at this scale
        correlation = np.max(result)
        if correlation > max_corr:
            max_corr = correlation
            best_result = result
            best_template_dim = (template_rescaled.shape[1], template_rescaled.shape[0])

    ij = np.unravel_index(np.argmax(best_result), best_result.shape)
    best_location = ij[::-1]

    return best_location, best_template_dim


def clean_image(image: np.ndarray) -> np.ndarray:
    thresh = filters.threshold_otsu(image)
    bw = morphology.closing(image > thresh, morphology.square(3))
    cleaned = segmentation.clear_border(bw)

    label_image = measure.label(cleaned)
    for region in measure.regionprops(label_image):
        if region.area >= 2000:
            box = [int(coord) for coord in resize_box(region.bbox, 1.05)]
            cleaned[box[0] : box[2], box[1] : box[3]] = 0

    cleaned = morphology.remove_small_objects(cleaned, 3)
    cleaned = np.invert(cleaned)

    return cleaned
