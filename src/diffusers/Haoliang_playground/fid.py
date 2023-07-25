from zipfile import ZipFile
import requests
import numpy as np


def download(url, local_filepath):
    r = requests.get(url)
    with open(local_filepath, "wb") as f:
        f.write(r.content)
    return local_filepath


dummy_dataset_url = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/sample-imagenet-images.zip"
local_filepath = download(dummy_dataset_url, dummy_dataset_url.split("/")[-1])

with ZipFile(local_filepath, "r") as zipper:
    zipper.extractall(".")
    
from PIL import Image
import os

dataset_path = "sample-imagenet-images"
image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])

real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]

from diffusers import StableDiffusionPipeline
import torch

model_ckpt = "CompVis/stable-diffusion-v1-4"
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
weight_dtype = torch.float16
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=weight_dtype).to(device)

prompts = [
    "cassette player",
    "chainsaw",
    "chainsaw",
    "church",
    "gas pump",
    "gas pump",
    "gas pump",
    "parachute",
    "parachute",
    "tench",
]

seed = 0
generator = torch.manual_seed(seed)

fake_images = sd_pipeline(prompts, num_images_per_prompt=1, generator=generator, output_type="numpy").images

fake_images = torch.tensor(fake_images)
fake_images = fake_images.permute(0, 3, 1, 2)

from torchmetrics.image.fid import FrechetInceptionDistance

fid = FrechetInceptionDistance(normalize=True)
fid.update(real_images, real=True)
fid.update(fake_images, real=False)

print(f"FID: {float(fid.compute())}")