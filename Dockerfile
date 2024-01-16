FROM python:3.10-slim-buster AS base

RUN pip install --no-cache-dir pdm
ADD ./pyproject.toml ./pdm.lock ./

FROM base AS builder

RUN apt-get update
RUN apt-get install -y unzip

# Download model.
RUN pip install gdown
RUN gdown 1IMwzqZUuRnTv5jcuKdvZx-RZweknww5x -O models.zip
RUN unzip "models.zip" && mv "09-11-2019 DCPv2 model" models
ADD https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth ./models/RealESRGAN_x4plus_anime_6B.pth

RUN pdm sync -G generate-onnx

# Build ONNX models.
ADD ./vendor/ ./vendor/
ADD ./generate-onnx.py ./
RUN pdm run generate-onnx.py

FROM base

RUN pdm sync -G server && pdm cache clear

COPY --from=builder ./vendor/ ./vendor/

ADD ./server.py ./decensor.py ./predict.py ./utils.py ./esrgan_runner.py ./

CMD ["pdm", "run", "uvicorn", \
	"--host", "0.0.0.0", "--port", "$PORT", \
	"server:app"]
