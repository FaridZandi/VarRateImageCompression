import os
import argparse

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, type=str, help='path to model')
parser.add_argument('--input', required=True, type=str, help='input codes')
parser.add_argument('--output', default='.', type=str, help='output folder')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--iterations', type=int, default=16, help='unroll iterations')
args = parser.parse_args()

def load_codes(content):
    codes = []
    for code in content["codes"]:
        patch_codes = np.unpackbits(code)
        patch_codes = np.reshape(patch_codes, (-1, 32, 2, 2)).astype(np.float32) * 2 - 1
        codes.append(torch.from_numpy(patch_codes)) 
    return codes

content = np.load(args.input)
codes = load_codes(content)
image_height = content["height"]
image_width = content["width"]
batch_size = len(codes)
channels = codes[0].shape[1]

# print(batch_size, channels, image_height, image_width)

height = 32 
width = 32 
patch_size = 32

with torch.no_grad():   
    for i in range(batch_size):
        codes[i] = Variable(codes[i])

import network

decoder = network.DecoderCell()
decoder.eval()

decoder.load_state_dict(torch.load(args.model))

with torch.no_grad():
    decoder_h_1 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)),
                   Variable(torch.zeros(batch_size, 512, height // 16, width // 16)))

    decoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)),
                   Variable(torch.zeros(batch_size, 512, height // 8, width // 8)))

    decoder_h_3 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)),
                   Variable( torch.zeros(batch_size, 256, height // 4, width // 4)))

    decoder_h_4 = (Variable(torch.zeros(batch_size, 128, height // 2, width // 2)),
                   Variable(torch.zeros(batch_size, 128, height // 2, width // 2)))


if args.cuda:
    decoder = decoder.cuda()

    decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
    decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
    decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
    decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())


padded_height = (image_height + 31) - ((image_height + 31) % 32)
padded_width = (image_width + 31) - ((image_width + 31) % 32)

image = torch.zeros((1, 3, padded_height, padded_width)) + 0.5
# print("image.shape", image.shape)

for iters in range(args.iterations):

    with torch.no_grad():

        iter_codes = []
        for i in range(batch_size):
            if codes[i].shape[0] > iters:
                iter_codes.append(codes[i][iters].unsqueeze(0))
            else:
                iter_codes.append(torch.zeros((1,32,2,2)))

        stacked = torch.tensor(np.stack(iter_codes).squeeze(1)).cuda()

        # print("stacked.shape",stacked.shape)

        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            stacked, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

        # print ("output.shape ", output.shape)

        for batch in range(batch_size):
            if codes[batch].shape[0] >= iters:

                x_patches = padded_width // patch_size
                y_patches = padded_height // patch_size

                i = batch // x_patches
                j = batch % x_patches

                start_x = i * patch_size
                start_y = j * patch_size
                end_x = (i+1) * patch_size
                end_y = (j+1) * patch_size
                # print("rect: ", start_x, end_x, start_y, end_y)

                # print("imagepatch", image[0,:, start_x:end_x, start_y:end_y].shape)
                # print("outputpatch", output[batch].data.cpu().shape)

                image[0,:, start_x:end_x, start_y:end_y] += output[batch].data.cpu()

saved_image = image[:, :, :image_height, :image_width]
imsave(args.output,
       np.squeeze(saved_image.numpy().clip(0, 1) * 255.0).astype(np.uint8)
       .transpose(1, 2, 0))
