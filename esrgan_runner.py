import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer


@dataclass
class ImageInput:
    input_img: cv2.typing.MatLike
    input_path: Optional[str]

    @classmethod
    def from_file(cls, input_path: str):
        input_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        return cls(input_path, input_img)

    @classmethod
    def from_bytes(cls, input_bytes: bytes, input_path: Optional[str]):
        input_img = cv2.imdecode(np.frombuffer(input_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return cls(input_path, input_img)


@dataclass
class ImageOutput:
    output_img: cv2.typing.MatLike
    input_path: Optional[str]

    def save(self, output_path: str):
        cv2.imwrite(output_path, self.output_img)

    def save_dir(self, output_dir: str):
        if not self.input_path:
            raise ValueError('input_path is not set')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(self.input_path))
        self.save(output_path)

    def to_bytes(self) -> bytes:
        return self.to_ndarray().tobytes()

    def to_input(self) -> ImageInput:
        return ImageInput(self.output_img, self.input_path)

    def to_ndarray(self) -> np.ndarray:
        ext = '.png'
        if self.input_path:
            _, ext = os.path.splitext(self.input_path)
        return cv2.imencode(ext, self.output_img)[1]


def select_model(model_name: str) -> Tuple[RRDBNet, int]:
    """
    return RRDBNet, netscale
    """
    if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4
    elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4), 4

    elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4), 4

    elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2), 2


@lru_cache(maxsize=1)
def get_real_esrgan(model_name: str, *, dni_weight=None, tile: int = 400, half: bool = True) -> RealESRGANer:
    model, net_scale = select_model(model_name)
    model_path = os.path.join('models', model_name + '.pth')
    tile_pad = 10
    pre_pad = 0

    # restorer
    up_sampler = RealESRGANer(
        scale=net_scale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half,
        gpu_id=None,
    )
    return up_sampler


def run_upscale(image_input: ImageInput, scale: float = 4.0, *, half: bool = True, tile: int = 400,
                model_name='RealESRGAN_x4plus_anime_6B') -> ImageOutput:
    up_sampler = get_real_esrgan(model_name, tile=tile, half=half)

    output_img, _ = up_sampler.enhance(image_input.input_img, outscale=scale)
    return ImageOutput(output_img, image_input.input_path)


def run_downscale(image_input: ImageInput, downscale: float = 4.0) -> ImageOutput:
    # INTER_AREA resize
    output_img = cv2.resize(
        image_input.input_img, dsize=(0, 0), fx=1 / downscale, fy=1 / downscale,
        interpolation=cv2.INTER_AREA,
    )
    return ImageOutput(output_img, image_input.input_path)


if __name__ == '__main__':
    # Download model to models/RealESRGAN_x4plus_anime_6B.pth
    # https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth

    scale = 4.0
    image = ImageInput.from_file('decensor_input_original/test.png')
    out = run_downscale(image, downscale=scale)
    out = run_upscale(out.to_input(), scale=scale)
    out.save_dir('decensor_output')
