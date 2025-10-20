import os
import argparse
import time

import tqdm
from PIL import Image
import torch
import numpy as np
from PIL.Image import Resampling

from data.base_dataset import get_transform
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2im


class PredictOptions(TestOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--input_path', type=str, default='datasets/LR.png',
                            help='path to input image')
        parser.add_argument('--output_dir', type=str, default='./results/predict', help='output directory')
        return parser


def crop_image(img, crop_size=256):
    width, height = img.size
    crops = []
    positions = []
    for i in range(0, height, crop_size):
        for j in range(0, width, crop_size):
            box = (j, i, j + crop_size, i + crop_size)
            crop = img.crop(box)
            crops.append(crop)
            positions.append((i // crop_size, j // crop_size))
    return crops, positions


def merge_images(crops, positions, original_size=(2048, 2048), target_size=(12288, 12288)):
    scale = target_size[0] / original_size[0]
    crop_size = int(256 * scale)

    merged_img = Image.new('RGB', target_size)

    for crop, (row, col) in zip(crops, positions):
        x = col * crop_size
        y = row * crop_size
        merged_img.paste(crop, (x, y))

    return merged_img


if __name__ == '__main__':
    opt = PredictOptions().parse()
    os.makedirs(opt.output_dir, exist_ok=True)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    img = Image.open(opt.input_path).convert('RGB')

    start_time = time.time()
    crops, positions = crop_image(img, 256)

    processed_crops = []

    for crop in tqdm.tqdm(crops):
        resized_crop = crop.resize((768, 768), Resampling.BILINEAR)

        transform = get_transform(opt, grayscale=(opt.input_nc == 1))
        tensor = transform(resized_crop).unsqueeze(0).to(model.device)

        opt.dataset_mode = 'single'
        opt.num_test = 1

        with torch.no_grad():
            dummy_B = torch.zeros_like(tensor)
            start_pre_time = time.time()
            model.set_input({'A': tensor, 'B': dummy_B, 'A_paths': [opt.input_path]})
            model.test()
            end_pre_time = time.time()
            print(f"Prediction time: {end_pre_time - start_pre_time:.6f} seconds")
            visuals = model.get_current_visuals()

        for label, im_data in visuals.items():
            if label == 'fake_B':
                im = tensor2im(im_data)
                output_img = Image.fromarray(im).resize((1536, 1536), Resampling.BILINEAR)
                processed_crops.append(output_img)

    final_img = merge_images(processed_crops, positions, (2048, 2048), (12288, 12288))
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

    base_name, extension = os.path.splitext(os.path.basename(opt.input_path))
    output_filename = f"{base_name}_final{extension}"
    output_path = os.path.join(opt.output_dir, output_filename)
    final_img.save(output_path)
