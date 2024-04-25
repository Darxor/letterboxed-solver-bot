import io

import numpy as np
import pytesseract
from skimage import feature, filters, measure, morphology, segmentation, transform
from skimage import io as skio

from .imageregion import ImageRegion, stack_region_images, sort_regions_clockwise
from .utils import array2image, imgarray2bytesio, timed

TEMPLATE_PATH = "templates/template3.png"
TEMPLATE = skio.imread(TEMPLATE_PATH, as_gray=True, plugin="imageio")


async def process_image(image: io.BytesIO):
    img = skio.imread(image, as_gray=True, plugin="imageio")
    template = TEMPLATE

    template_loc, template_dim = find_template(img, template)
    img_cropped = crop_image(img, template_loc, template_dim, 1.5)
    img_cleaned, _, letter_regions = extract_image_regions(img_cropped)

    if not letter_regions:
        return imgarray2bytesio(img_cleaned), ""

    text_ocr = ocr_letters(letter_regions)
    text_ocr = (
        f"{text_ocr[:3]}\n{text_ocr[3:6]}\n{text_ocr[6:9]}\n{text_ocr[9:]}"
        if len(text_ocr) > 9
        else text_ocr
    )

    outimage = stack_region_images(letter_regions)
    return imgarray2bytesio(outimage), text_ocr

@timed
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

@timed
def find_template(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    min_scale = 0.2
    max_scale = 2
    num_levels = 10

    # find safe range of scales:
    # 1. template should not be larger than image
    # 2. template should not be smaller than 10% of the image,
    #    as we probably would not be able to detect letters at this scale anyway
    image_template_ratio = min(
        image.shape[0] / template.shape[0], image.shape[1] / template.shape[1]
    )
    min_scale = max(min_scale, 0.1 * image_template_ratio)
    max_scale = min(max_scale, image_template_ratio)
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=num_levels)

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

@timed
def extract_image_regions(
    image: np.ndarray,
) -> tuple[np.ndarray, list[ImageRegion], list[ImageRegion]]:
    thresh = filters.threshold_otsu(image)
    bw = morphology.closing(image > thresh, morphology.square(3))
    cleaned = segmentation.clear_border(bw)

    label_image = measure.label(cleaned)

    regions = [ImageRegion(region.bbox) for region in measure.regionprops(label_image)]

    max_area = max(r.area for r in regions)
    min_area = min(r.area for r in regions)

    # big regions are regions that are 85% of the biggest region
    # we assume that big regions are the regions that contain central field
    big_regions = [r for r in regions if r.area >= max_area * 0.85]
    
    # filter out regions that are too small or too big
    # and intersect with big regions
    letter_regions = [
        r
        for r in regions
        if min_area * 1.05 <= r.area < max_area * 0.85
        and not any(r.intersects(big_r) for big_r in big_regions)
    ]
    
    # sort letter regions clockwise starting from the top left corner
    # this will ensure correct order for solving the puzzle (given that we recognize letters correctly)
    letter_regions = sort_regions_clockwise(letter_regions, 45)
    
    cleaned = np.invert(cleaned)
    for region in letter_regions:
        region.rescale_bbox(1.15)
        region.square_bbox()
        region.crop_from_image(cleaned)

    return cleaned, big_regions, letter_regions

@timed
def ocr_letters(regions: list[ImageRegion]) -> str:
    ocr_letters = []

    for region in regions:
        letter = pytesseract.image_to_string(
            array2image(region.image),
            lang="eng",
            config="--psm 10 -c tessedit_char_whitelist=|lckmopsuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01",
        )
        ocr_letters.append(letter)

    text_replacers = {
        # look-alike characters
        "l": "I",
        "1": "I",
        "|": "I",
        "0": "O",
        # remove newlines and spaces
        "\n": "",
        " ": "",
    }

    text = "".join(letter[0] for letter in ocr_letters)
    for key, value in text_replacers.items():
        text = text.replace(key, value)

    return text.upper()
