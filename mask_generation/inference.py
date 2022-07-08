from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import cv2
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model import SODModel
from src.dataloader import InfDataloader, SODLoader
import torchvision.transforms as transforms


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    parser.add_argument('--img_path', default='./data/DUTS/DUTS-TE/DUTS-TE-Image', help='Path to folder containing images', type=str)
    parser.add_argument('--mask_path', default='./data/DUTS/DUTS-TE/DUTS-TE-Image', help='Path to folder containing images', type=str)
    parser.add_argument('--blurred_path', default='./data/DUTS/DUTS-TE/DUTS-TE-Image', help='Path to folder containing images', type=str)
    parser.add_argument('--model_path', default='/home/parsap/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection/models/0.7_wbce_w0-1_w1-1.12/best-model_epoch-204_mae-0.0505_loss-0.1370.pth', help='Path to model', type=str)
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)

    return parser.parse_args()


def run_inference(args):
    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel()
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    with torch.no_grad():

        img = cv2.imread(args.img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_np = torch.from_numpy(img).unsqueeze(0)

        # Pad images to target size
        img_tor = img.astype(np.float32)
        img_tor = img_tor / 255.0
        img_tor = np.transpose(img_tor, axes=(2, 0, 1))
        img_tor = torch.from_numpy(img_tor).float()
        t = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        img_tor = t(img_tor).unsqueeze(0)
        max_dim = 1024
        
        img_tor2 = torch.zeros((1,3,max_dim,max_dim)).cuda()
        img_tor2[0, :, :img_tor.shape[2], :img_tor.shape[3]] = img_tor

        # print(img_np.shape, img_tor2.shape)

        img_tor = img_tor.to(device)
        pred_masks2, _ = model(img_tor2)
        pred_masks = pred_masks2[:, :, :img_tor.shape[2], :img_tor.shape[3]]
        # print(pred_masks.shape)

        # Assuming batch_size = 1
        img_np = np.squeeze(img_np.numpy(), axis=0)
        img_np = img_np.astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # print(pred_masks.cpu().numpy().shape)
        pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
        pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))

        directory='results/'
        jpg_directory='jpg_blurred/'

        img_dentry = args.img_path.split("/")[-1]
        path = "/".join(args.img_path.split("/")[:-1])
        name = img_dentry.split(".")[0]
        ext = img_dentry.split(".")[1]

        
        # filename_mask = path + "/" + name + "_mask" + ".png"
        # filename_jpg = path + "/" + name + "_blurred" + ".jpg"

        # cv2.imwrite(filename_original_image, img_np)
        cv2.imwrite(args.mask_path, pred_masks_raw * 255)

        three_channle_mask = np.repeat(
            pred_masks_round.reshape(img_tor.shape[2], img_tor.shape[3], 1), 
            3, axis=2
        )

        kernel = np.ones((5, 5), np.float32) / 25
        blurred_whole_image = cv2.filter2D(img_np, -1, kernel)

        blurred_background = blurred_whole_image * (1 - three_channle_mask)

        final_image = blurred_background + (img_np * three_channle_mask)

        cv2.imwrite(args.blurred_path, final_image, [cv2.IMWRITE_JPEG_QUALITY, 50])

if __name__ == '__main__':
    rt_args = parse_arguments()
    run_inference(rt_args)
