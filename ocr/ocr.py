import io
import pytesseract
import cv2 as cv
import numpy as np

TEMPLATE_PATH = "templates/template3.png"


def read_image_stream(image_stream: io.BytesIO) -> np.ndarray:
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    return cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)


async def process_image(image: io.BytesIO):
    img = read_image_stream(image)

    template = cv.imread(TEMPLATE_PATH, cv.IMREAD_GRAYSCALE)

    if not template.any():
        raise FileNotFoundError(f"Template file could not be read: {TEMPLATE_PATH}")

    w, h = template.shape[::-1]

    cols, rows = img.shape
    brightness = np.sum(img) / (255 * cols * rows)
    minimum_brightness = 0.5

    ratio = brightness / minimum_brightness
    if ratio != 1:
        img = cv.convertScaleAbs(img, alpha=1 / ratio, beta=0)

    # img = cv.blur(img,(3,3))

    method = cv.TM_CCOEFF
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    img[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = (
        255
    )

    expand_factor = int(w * 0.25)
    top_left = (top_left[0] - expand_factor, top_left[1] - expand_factor)
    bottom_right = (bottom_right[0] + expand_factor, bottom_right[1] + expand_factor)
    cropped_img = img[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]

    # upper_side = cropped_img[0:int(w*0.3),]
    _, thresh = cv.threshold(cropped_img, 150, 255, cv.THRESH_BINARY)
    # _, thresh = cv.threshold(upper_side, 185, 255, cv.THRESH_BINARY)
    
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv.dilate(thresh, kernel, iterations=2)
    black = cv.bitwise_not(dilated)

    text = pytesseract.image_to_string(
        black,
        lang="eng",
        config="-c tessedit_char_whitelist=lckmopsuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01 --psm 6",
    )
    text_ocr = (
        text.strip().replace("0", "O").replace("1", "I").replace("l", "I").upper()
    )

    _, outimage_buffer = cv.imencode(".png", black)

    return io.BytesIO(outimage_buffer), text_ocr
