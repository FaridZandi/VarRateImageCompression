import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 
from argparse import Namespace
from os.path import exists
from pathlib import Path
from PIL import Image
from scipy.misc import imread, imresize, imsave
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
import argparse
import dataset
import json
import numpy as np
import os
import random
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.utils.data as data
import yaml
import csv   
import imagesize
import traceback
import logging
import pandas as pd
import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
from PIL import Image



def get_bpp(row):
    pixels = row["width"] * row["height"]
    size = row["file_size"]
    return size / pixels * 8


def psnr(original_image_path, compared_image_path):
    if isinstance(original_image_path, str):
        original = np.array(Image.open(original_image_path).convert('RGB'), dtype=np.float32)
    if isinstance(compared_image_path, str):
        compared = np.array(Image.open(compared_image_path).convert('RGB'), dtype=np.float32)

    mse = np.mean(np.square(original - compared))
    psnr = np.clip(
        np.multiply(np.log10(255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)[0]
    return psnr



def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    return g / g.sum()







def _SSIMForMultiScale(img1,
                       img2,
                       max_val=255,
                       filter_size=11,
                       filter_sigma=1.5,
                       k1=0.01,
                       k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.
  This function attempts to match the functionality of ssim_index_new.m by
  Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
  Returns:
    Pair containing the mean SSIM and contrast sensitivity between `img1` and
    `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).', img1.shape,
            img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)))
    cs = np.mean(v1 / v2)
    return ssim, cs


def MultiScaleSSIM(img1,
                   img2,
                   max_val=255,
                   filter_size=11,
                   filter_sigma=1.5,
                   k1=0.01,
                   k2=0.03,
                   weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.
  This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
  Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
  similarity for image quality assessment" (2003).
  Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
  Author's MATLAB implementation:
  http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  Arguments:
    img1: Numpy array holding the first RGB image batch.
    img2: Numpy array holding the second RGB image batch.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    filter_size: Size of blur kernel to use (will be reduced for small images).
    filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
      for small images).
    k1: Constant used to maintain stability in the SSIM calculation (0.01 in
      the original paper).
    k2: Constant used to maintain stability in the SSIM calculation (0.03 in
      the original paper).
    weights: List of weights for each level; if none, use five levels and the
      weights from the original paper.
  Returns:
    MS-SSIM score between `img1` and `img2`.
  Raises:
    RuntimeError: If input images don't have the same shape or don't have four
      dimensions: [batch_size, height, width, depth].
  """
    if img1.shape != img2.shape:
        raise RuntimeError(
            'Input images must have the same shape (%s vs. %s).', img1.shape,
            img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else
                       [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float64) for x in [img1, img2]]
    mssim = np.array([])
    mcs = np.array([])
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
            im1,
            im2,
            max_val=max_val,
            filter_size=filter_size,
            filter_sigma=filter_sigma,
            k1=k1,
            k2=k2)
        mssim = np.append(mssim, ssim)
        mcs = np.append(mcs, cs)
        filtered = [
            convolve(im, downsample_filter, mode='reflect')
            for im in [im1, im2]
        ]
        im1, im2 = [x[:, ::2, ::2, :] for x in filtered]
    return (np.prod(mcs[0:levels - 1]**weights[0:levels - 1]) *
            (mssim[levels - 1]**weights[levels - 1]))


def msssim(original, compared):
    if isinstance(original, str):
        original = np.array(Image.open(original).convert('RGB'), dtype=np.float32)
    if isinstance(compared, str):
        compared = np.array(Image.open(compared).convert('RGB'), dtype=np.float32)

    original = original[None, ...] if original.ndim == 3 else original
    compared = compared[None, ...] if compared.ndim == 3 else compared

    return MultiScaleSSIM(original, compared, max_val=255)



parser = argparse.ArgumentParser()
parser.add_argument("--output", default="metrics_out.csv", type=str)
args = parser.parse_args()


fieldnames = ["name", "image_class", "file_size", "width", "height", "setting", 
              "classifier", "top1", "top2", "top3", 
              "top4", "top5", "ground_truth_percentage", "file_path"]


all_files = [
    # "result_parr_1.csv", 
    # "result_parr_2.csv", 
    # "result_parr_3.csv",
    # "result_parr_4.csv",
    "results_large.csv"
]    

df_from_each_file = (pd.read_csv(f, names=fieldnames) for f in all_files)
df = pd.concat(df_from_each_file, ignore_index=True)
df = df.drop_duplicates()

images_df = df[df["setting"] == "original"].groupby(["file_path", "image_class"]).size().reset_index().rename(columns={0:'count'})[["file_path", "image_class"]]

images = [tuple(x) for x in images_df.to_numpy()]

print(images)


df = df.drop(['file_path'], axis=1)
df["bpp"] = df.apply(lambda x: get_bpp(x), axis=1)


def record_metrics(original, compared, code, name, img_class, setting):
    this_psnr = psnr(original, compared)
    this_msssim = msssim(original, compared)

    width, height = imagesize.get(original)

    with open(args.output, 'a+') as f:
        fieldnames = ["name", "img_class", "setting", "height", "width", "psnr", "msssim", "bpp", "original_path", "file_path"]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writerow({
            "name": name,
            "img_class": img_class, 
            "width": width, 
            "height": height, 
            "bpp": 8 * int(os.path.getsize(code)) / (width * height),
            "setting": setting, 
            "psnr": this_psnr, 
            "msssim": this_msssim,
            "original_path": original,
            "file_path": compared,
        })

# images = [
#     ("/u/faridzandi/csc2231/imagenet/temp/imagenet-mini/train/n01443537/n01443537_20131.JPEG", 1),
#     ("/u/faridzandi/csc2231/imagenet/temp/imagenet-mini/train/n01484850/n01484850_17712.JPEG", 2),
#     ("/u/faridzandi/csc2231/imagenet/temp/imagenet-mini/train/n01484850/n01484850_31115.JPEG", 3)
# ]

for img_path, img_class in images:
    try: 
        print("processing:", img_path)
     
        img_dentry = img_path.split("/")[-1]
        path = "/".join(img_path.split("/")[:-1])
        name = img_dentry.split(".")[0]
        ext = img_dentry.split(".")[1] 

        mask_path = path + "/" + name + "_mask" + ".png"
        blurred_path = path + "/" + name + "_blurred" + ".jpg"

        record_metrics(img_path, blurred_path, blurred_path, name, img_class, "blurred")

        for encoder_mapping in range (1,8): 
            if encoder_mapping == 1:
                for quality in range (16):
                    suffix = str(encoder_mapping) + "_" + str(quality) 
                    code_path_noext = path + "/" + name + "_compressed_" + suffix
                    code_path = path + "/" + name + "_compressed_" + suffix + ".npz"
                    decoded_path = path + "/" + name + "_decoded_" + suffix + ".jpg"

                    record_metrics(img_path, decoded_path, code_path, name, img_class, "code_" + suffix)

            else:
                suffix = str(encoder_mapping)
                code_path_noext = path + "/" + name + "_compressed_" + suffix
                code_path = path + "/" + name + "_compressed_" + suffix + ".npz"
                decoded_path = path + "/" + name + "_decoded_" + suffix + ".jpg"

                record_metrics(img_path, decoded_path, code_path, name, img_class, "code_" + suffix)


    except Exception as e:  
        logging.error(traceback.format_exc())
