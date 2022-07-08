## some function borrowed from
## https://github.com/tensorflow/models/blob/master/compression/image_encoder/msssim.py
"""Python implementation of MS-SSIM.

Usage:

python msssim.py --original_image=original.png --compared_image=distorted.png
"""
import argparse

import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--metric', '-m', type=str, default='all', help='metric')
parser.add_argument(
    '--original-image', '-o', type=str, required=True, help='original image')
parser.add_argument(
    '--compared-image', '-c', type=str, required=True, help='compared image')
args = parser.parse_args()




def main():
    if args.metric != 'psnr':
        print(msssim(args.original_image, args.compared_image), end='')
    if args.metric != 'ssim':
        print(psnr(args.original_image, args.compared_image), end='')


if __name__ == '__main__':
    main()
