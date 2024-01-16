import io
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable

from PIL import Image
from tqdm import tqdm

from decensor import decensor  # Assuming decensor is a function you have defined or imported from a module

base_dir = r''

decensor_input_path = os.path.join(base_dir, 'decensor_input')
decensor_input_original_path = os.path.join(base_dir, 'decensor_input_original')
decensor_output_path = os.path.join(base_dir, 'decensor_output')

os.makedirs(decensor_output_path, exist_ok=True)


def filter_dot_begin(iterable: Iterable[str]) -> Iterable[str]:
    return filter(lambda filename: not filename.startswith('.'), iterable)


def run(decensor_input, decensor_input_original, decensor_output, *, variant_number=0, worker_count=3):
    # List files in the input directories
    input_files = set(filter_dot_begin(os.listdir(decensor_input)))
    input_original_files = set(filter_dot_begin(os.listdir(decensor_input_original)))

    # List files in the output directory
    output_files = set(os.listdir(decensor_output))

    # Find common files in both input directories
    common_files = list(input_files.intersection(input_original_files))
    common_files.sort()

    # Create a list of tasks to process
    tasks = []
    for file_name in common_files:
        # Skip if the file already exists in the output directory
        if file_name in output_files:
            continue

        input_path = os.path.join(decensor_input, file_name)
        input_original_path = os.path.join(decensor_input_original, file_name)
        output_path = decensor_output

        tasks.append((input_original_path, input_path, output_path, variant_number))

    # Process tasks concurrently and handle exceptions
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        # Submit tasks to the executor
        future_to_task = {executor.submit(decensor_local, *task): task for task in tasks}

        # Process completed tasks
        for future in tqdm(as_completed(future_to_task), total=len(tasks)):
            try:
                # Attempt to get the result, which can raise an exception
                future.result()
            except Exception as exc:
                task = future_to_task[future]
                filename = os.path.basename(task[0])  # Extract the input original path (file name)
                print(f"An error occurred with file '{filename}': {exc}")
                executor.shutdown(wait=False, cancel_futures=True)
                raise exc


def decensor_local(input_original_path: str, input_path: str, output_path: str, variant_number=0,
                   rename_with_variation=False):
    # Load the original and censored images using PIL.Image
    with open(input_original_path, 'rb') as file:
        original_img = Image.open(io.BytesIO(file.read()))

    with open(input_path, 'rb') as file:
        censored_img = Image.open(io.BytesIO(file.read()))

    # Apply variant transformations
    original_img = apply_variant(original_img, variant_number)
    censored_img = apply_variant(censored_img, variant_number)

    # Apply the decensor function
    img = decensor(original_img, censored_img, is_mosaic=True)

    # Revert the transformation on the final image
    img = revert_variant(img, variant_number)

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Construct the output file path
    output_file_path = os.path.join(output_path, os.path.basename(input_original_path))
    if rename_with_variation:
        name, ext = os.path.splitext(os.path.basename(input_original_path))
        output_file_path = os.path.join(output_path, f'{name} {variant_number}{ext}')
    tmp_output_file_path = os.path.join(output_path, '.' + os.path.basename(input_original_path))

    # Extract the image format from the original file path
    img_format = os.path.splitext(input_original_path)[1][1:].upper()
    if img_format == 'JPG':
        img_format = 'JPEG'

    # Save the image in the same format as the original
    if img_format == 'JPEG':
        img.save(tmp_output_file_path, format=img_format, quality=98, optimize=True)
    else:
        # For formats like PNG, where quality parameter is not used
        img.save(tmp_output_file_path)
    if os.path.isfile(output_file_path):
        os.remove(output_file_path)
    os.rename(tmp_output_file_path, output_file_path)


def apply_variant(image, variant_number):
    if variant_number == 0:
        return image
    elif variant_number == 1:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif variant_number == 2:
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return image.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)


def revert_variant(image, variant_number):
    # Revert the transformations applied by apply_variant
    if variant_number == 0:
        return image
    elif variant_number == 1:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    elif variant_number == 2:
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        return image.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)


def run_all_variantion(decensor_input, decensor_input_original, decensor_output, *, worker_count=3):
    # List files in the input directories
    input_files = set(filter_dot_begin(os.listdir(decensor_input)))
    input_original_files = set(filter_dot_begin(os.listdir(decensor_input_original)))

    # List files in the output directory
    output_files = set(os.listdir(decensor_output))

    # Find common files in both input directories
    common_files = list(input_files.intersection(input_original_files))
    common_files.sort()

    # Create a list of tasks to process
    tasks = []
    for file_name in common_files:
        # Skip if the file already exists in the output directory
        if file_name in output_files:
            continue

        input_path = os.path.join(decensor_input, file_name)
        input_original_path = os.path.join(decensor_input_original, file_name)
        output_path = decensor_output
        for variant_number in range(4):
            tasks.append((input_original_path, input_path, output_path, variant_number, True))

    # Process tasks concurrently and handle exceptions
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        # Submit tasks to the executor
        future_to_task = {executor.submit(decensor_local, *task): task for task in tasks}

        # Process completed tasks
        for future in tqdm(as_completed(future_to_task), total=len(tasks)):
            try:
                # Attempt to get the result, which can raise an exception
                future.result()
            except Exception as exc:
                task = future_to_task[future]
                filename = os.path.basename(task[0])  # Extract the input original path (file name)
                print(f"An error occurred with file '{filename}': {exc}")
                executor.shutdown(wait=False, cancel_futures=True)
                raise exc


if __name__ == '__main__':
    # run_all_variantion(decensor_input_path, decensor_input_original_path, decensor_output_path, worker_count=2)
    while 1:
        run(decensor_input_path, decensor_input_original_path, decensor_output_path, variant_number=0, worker_count=4)
        time.sleep(5)
