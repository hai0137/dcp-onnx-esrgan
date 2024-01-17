"""
Cleaned from: DeepCreamPy/decensor.py
"""
import logging
from multiprocessing.pool import ThreadPool
from typing import Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import measurements

from predict import predict
from utils import image_to_array, expand_bounding

# Green.
MASK_COLOR = [0, 1, 0]

logger = logging.getLogger('decensor')


def find_mask(colored):
    mask = np.ones(colored.shape, np.uint8)
    i, j = np.where(np.all(colored[0] == MASK_COLOR, axis=-1))
    mask[0, i, j] = 0
    return mask


# Performant connected-components algorithm.
def find_regions(image, mask_color):
    pixels = np.array(image)
    array = np.all(pixels == mask_color, axis=2)
    labeled, n_components = measurements.label(array)
    indices = np.moveaxis(np.indices(array.shape), 0, -1)[:, :, [1, 0]]

    regions = []
    for index in range(1, n_components + 1):
        regions.append(indices[labeled == index].tolist())
    regions.sort(key=len, reverse=True)
    return regions


def resize(img_array: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    pred_img = Image.fromarray(img_array.astype('uint8'))
    pred_img = pred_img.resize(size, resample=Image.BICUBIC)
    return image_to_array(pred_img)


def resize_upscale(img_array: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    import cv2

    from esrgan_runner import ImageInput, run_upscale

    # Convert from BGR to RGB, resize, and then convert back to BGR
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    target_width, target_height = size
    origin_height, origin_width = img_array.shape[:2]  # (height, width)
    scale = target_height / origin_height
    img = ImageInput(img_array, None)
    out = run_upscale(img, scale)
    out_img = out.output_img

    out_height, out_width = out_img.shape[:2]
    if abs(out_width - target_width) > 1:
        out_img = cv2.resize(
            out_img, dsize=(0, 0), fx=target_width / out_width, fy=1,
            interpolation=cv2.INTER_AREA,
        )

    # convert back to BGR
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    return image_to_array(out_img)


def resize_down_up_scale(img_array: np.ndarray, size: Tuple[int, int], downscale=5.0) -> np.ndarray:
    import cv2

    from esrgan_runner import ImageInput, run_upscale, run_downscale

    # Convert from BGR to RGB, resize, and then convert back to BGR
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    target_width, target_height = size

    img = ImageInput(img_array, None)
    # Downscale first
    down_img = run_downscale(img, downscale)

    if downscale > 4.0:  # middle upscale
        down_img = run_upscale(down_img.to_input(), 4)

    down_img_height, _ = down_img.output_img.shape[:2]

    # Upscale
    scale = target_height / down_img_height
    out = run_upscale(down_img.to_input(), scale)
    out_img = out.output_img

    out_height, out_width = out_img.shape[:2]
    if abs(out_width - target_width) > 1:
        out_img = cv2.resize(
            out_img, dsize=(0, 0), fx=target_width / out_width, fy=1,
            interpolation=cv2.INTER_AREA,
        )

    # convert back to BGR
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    return image_to_array(out_img)


def decensor(original_image: Image, colored: Image, is_mosaic: bool) -> Image:
    ori = original_image
    # save the alpha channel if the image has an alpha channel
    alpha_channel = None
    has_alpha = False
    if ori.mode == "RGBA":
        has_alpha = True
        alpha_channel = np.asarray(ori)[:, :, 3]
        alpha_channel = np.expand_dims(alpha_channel, axis=-1)
        ori = ori.convert('RGB')

    ori_array = image_to_array(ori)
    if is_mosaic:
        # if mosaic decensor, mask is empty
        colored = colored.convert('RGB')
        color_array = image_to_array(colored)
        color_array = np.expand_dims(color_array, axis=0)
        mask = find_mask(color_array)
    else:
        ori_array_mask = np.expand_dims(ori_array, axis=0)
        mask = find_mask(ori_array_mask)

    # colored image is only used for finding the regions
    regions = find_regions(colored.convert('RGB'), [v * 255 for v in MASK_COLOR])
    logger.debug(f"Found {len(regions)} censored regions in this image!")
    if len(regions) == 0 and not is_mosaic:
        logger.info("No green (0,255,0) regions detected! Make sure you're using exactly the right color.")
        return ori

    def predict_region(region):
        bounding_box = expand_bounding(ori, region, expand_factor=1.5)
        crop_img = ori.crop(bounding_box)

        # convert mask back to image
        mask_reshaped = mask[0, :, :, :] * 255.0
        mask_img = Image.fromarray(mask_reshaped.astype('uint8'))
        # resize the cropped images

        crop_img = crop_img.resize((256, 256), resample=Image.NEAREST)
        crop_img_array = image_to_array(crop_img)

        # resize the mask images
        mask_img = mask_img.crop(bounding_box)
        mask_img = mask_img.resize((256, 256), resample=Image.NEAREST)

        # convert mask_img back to array
        mask_array = image_to_array(mask_img)
        # the mask has been upscaled so there will be values not equal to 0 or 1

        if not is_mosaic:
            a, b = np.where(np.all(mask_array == 0, axis=-1))
            crop_img_array[a, b, :] = 0.

        # Normalize.
        crop_img_array = crop_img_array * 2.0 - 1

        # Queue prediction request.
        pred_img_array = predict(crop_img_array, mask_array, is_mosaic)
        pred_img_array = (255.0 * ((pred_img_array + 1.0) / 2.0)).astype(np.uint8)
        return pred_img_array, bounding_box

    # Run predictions.
    with ThreadPool() as pool:
        results = pool.map(predict_region, regions)

    output_img_array = ori_array.copy()
    for (pred_img_array, bounding_box), region in zip(results, regions):
        # scale prediction image back to original size
        bounding_width = bounding_box[2] - bounding_box[0]
        bounding_height = bounding_box[3] - bounding_box[1]

        # convert np array to image
        # pred_img_array = resize(img_array=pred_img_array.astype('uint8'), size=(bounding_width, bounding_height))
        pred_img_array = resize_upscale(img_array=pred_img_array.astype('uint8'),
                                        size=(bounding_width, bounding_height))

        # Efficiently copy regions into output image.
        for (x, y) in region:
            if bounding_box[0] <= x < bounding_box[0] + bounding_width:
                if bounding_box[1] <= y < bounding_box[1] + bounding_height:
                    output_img_array[y][x] = pred_img_array[y - bounding_box[1]][x - bounding_box[0]]

    output_img_array = output_img_array * 255.0

    # restore the alpha channel if the image had one
    if has_alpha:
        output_img_array = np.concatenate((output_img_array, alpha_channel), axis=2)

    merged_img = Image.fromarray(output_img_array.astype('uint8'))
    merged_img.info = original_image.info
    return merged_img
