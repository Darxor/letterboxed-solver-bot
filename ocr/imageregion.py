import math
from functools import cached_property

import numpy as np

from .utils import bounding_square, rescale_box


class ImageRegion:
    def __init__(
        self,
        bbox: tuple[int, int, int, int],
        overscale: float = 1.0,
        square_bbox: bool = False,
    ):
        self.bbox = bbox
        if square_bbox:
            self.square_bbox()
        self.rescale_bbox(overscale)
        self.image = None

    def square_bbox(self) -> None:
        self.bbox = bounding_square(self.bbox)

    def rescale_bbox(self, factor: float) -> None:
        self.bbox = rescale_box(self.bbox, factor)

    @cached_property
    def shape(self) -> tuple[int, int]:
        minr, minc, maxr, maxc = self.bbox
        return maxr - minr, maxc - minc

    @cached_property
    def centroid(self) -> tuple[float, float]:
        minr, minc, maxr, maxc = self.bbox
        return (minr + maxr) / 2, (minc + maxc) / 2

    @cached_property
    def area(self) -> int:
        height, width = self.shape
        return height * width

    def crop_from_image(self, source_image: np.ndarray) -> None:
        self.image = source_image[
            self.bbox[0] : self.bbox[2], self.bbox[1] : self.bbox[3]
        ]

    def intersects(self, other: "ImageRegion") -> bool:
        minr1, minc1, maxr1, maxc1 = self.bbox
        minr2, minc2, maxr2, maxc2 = other.bbox

        return not (minr1 > maxr2 or maxr1 < minr2 or minc1 > maxc2 or maxc1 < minc2)


def stack_region_images(regions: list[ImageRegion]) -> np.ndarray:
    """
    Stack images of regions horizontally
    """
    images = [region.image.copy() for region in regions]
    maxdims = np.max([img.shape for img in images], axis=0)

    for i, img in enumerate(images):
        new_img = np.pad(
            img,
            ((0, maxdims[0] - img.shape[0]), (0, maxdims[1] - img.shape[1])),
            mode="constant",
            constant_values=1,
        )
        images[i] = new_img

    return np.hstack(images)


def sort_regions_clockwise(
    regions: list[ImageRegion], start_angle_deg: int = 90
) -> list[ImageRegion]:
    start_angle_rad = start_angle_deg * (math.pi / 180)  # Convert degrees to radians

    def find_pivot():
        x_mean = sum([pt.centroid[0] for pt in regions]) / len(regions)
        y_mean = sum([pt.centroid[1] for pt in regions]) / len(regions)
        return (x_mean, y_mean)

    pivot = find_pivot()

    def angle_from_pivot(box):
        # Calculate the angle, adjust by the start angle, and negate for clockwise order
        angle = math.atan2(box.centroid[0] - pivot[0], box.centroid[1] - pivot[1])
        adjusted_angle = angle - start_angle_rad
        # Normalize the angle to keep it in the -pi to pi range
        if adjusted_angle < -math.pi:
            adjusted_angle += 2 * math.pi
        elif adjusted_angle > math.pi:
            adjusted_angle -= 2 * math.pi
        return adjusted_angle

    return sorted(regions, key=angle_from_pivot)
