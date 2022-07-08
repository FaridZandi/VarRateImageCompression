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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to train your model.')
    parser.add_argument('--imgs_folder', default='./data/DUTS/DUTS-TE/DUTS-TE-Image', help='Path to folder containing images', type=str)
    parser.add_argument('--model_path', default='/home/parsap/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection/models/0.7_wbce_w0-1_w1-1.12/best-model_epoch-204_mae-0.0505_loss-0.1370.pth', help='Path to model', type=str)
    parser.add_argument('--use_gpu', default=True, help='Whether to use GPU or not', type=bool)
    parser.add_argument('--img_size', default=256, help='Image size to be used', type=int)
    parser.add_argument('--bs', default=24, help='Batch Size for testing', type=int)

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

    inf_data = InfDataloader(img_folder=args.imgs_folder, target_size=args.img_size)
    # Since the images would be displayed to the user, the batch_size is set to 1
    # Code at later point is also written assuming batch_size = 1, so do not change
    inf_dataloader = DataLoader(inf_data, batch_size=1, shuffle=True, num_workers=2)

    print("Press 'q' to quit.")
    with torch.no_grad():
        for batch_idx, (img_np, img_tor) in enumerate(inf_dataloader, start=1):
            print(img_np.shape, img_tor.shape)
            img_tor = img_tor.to(device)
            pred_masks, _ = model(img_tor)

            # Assuming batch_size = 1
            img_np = np.squeeze(img_np.numpy(), axis=0)
            img_np = img_np.astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            pred_masks_raw = np.squeeze(pred_masks.cpu().numpy(), axis=(0, 1))
            pred_masks_round = np.squeeze(pred_masks.round().cpu().numpy(), axis=(0, 1))
            print('Image :', batch_idx)
            directory='results/'
            jpg_directory='jpg_blurred/'
            filename_original_image = str(batch_idx)+'_original.png'
            filename_mask=str(batch_idx)+'_mask.png'
            filename_jpg=str(batch_idx)+'_jpg_mask.jpg'

            cv2.imwrite(directory+filename_original_image,img_np)
            cv2.imwrite(directory+filename_mask, pred_masks_raw*255)



            three_channle_mask=np.repeat(pred_masks_round.reshape(256, 256, 1), 3, axis=2)

            kernel =np.ones((5,5),np.float32)/25
            blurred_whole_image = cv2.filter2D(img_np,-1,kernel)

            blurred_background= blurred_whole_image*(1-three_channle_mask)

            final_image=blurred_background+ (img_np*three_channle_mask)


            cv2.imwrite(jpg_directory+filename_jpg,final_image,[cv2.IMWRITE_JPEG_QUALITY, 50])


           # cv2.imshow('Input Image', img_np)
           # cv2.imshow('Generated Saliency Mask', pred_masks_raw)
           # cv2.imshow('Rounded-off Saliency Mask', pred_masks_round)

            key = cv2.waitKey(0)
            if key == ord('q'):
                break


def calculate_mae(args):
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

    test_data = SODLoader(mode='test', augment_data=False, target_size=args.img_size)
    test_dataloader = DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=2)

    # List to save mean absolute error of each image
    mae_list = []
    with torch.no_grad():
        for batch_idx, (inp_imgs, gt_masks) in enumerate(tqdm.tqdm(test_dataloader), start=1):
            inp_imgs = inp_imgs.to(device)
            gt_masks = gt_masks.to(device)
            pred_masks, _ = model(inp_imgs)

            mae = torch.mean(torch.abs(pred_masks - gt_masks), dim=(1, 2, 3)).cpu().numpy()
            mae_list.extend(mae)

    print('MAE for the test set is :', np.mean(mae_list))


if __name__ == '__main__':
    rt_args = parse_arguments()
    run_inference(rt_args)
