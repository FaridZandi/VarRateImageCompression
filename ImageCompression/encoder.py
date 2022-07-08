import argparse

import numpy as np
from scipy.misc import imread, imresize, imsave

import torch
from torch.autograd import Variable

import os 


parser = argparse.ArgumentParser()
parser.add_argument(
    '--model', '-m', required=True, type=str, help='path to model')
parser.add_argument(
    '--input', '-i', required=True, type=str, help='input image')
parser.add_argument(
    '--mask', '-k', required=True, type=str, help='input mask')
parser.add_argument(
    '--mapping', '-p', required=True, type=int, help='roi to quality mapping')
parser.add_argument(
    '--quality', '-q', default=16, type=int, help='fixed quality')
parser.add_argument(
    '--output', '-o', required=True, type=str, help='output codes')
parser.add_argument('--cuda', '-g', action='store_true', help='enables cuda')
parser.add_argument(
    '--iterations', type=int, default=16, help='unroll iterations')
args = parser.parse_args()

image = imread(args.input, mode='RGB')
image = torch.from_numpy(
    np.expand_dims(
        np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1)), 0))

mask = imread(args.mask, mode='L')
mask = torch.from_numpy(
    np.expand_dims(mask.astype(np.float32) / 255.0, 0))
mask = mask.unsqueeze(1)


batch_size, input_channels, image_height, image_width = image.size()

padded_height = (image_height + 31) - ((image_height + 31) % 32)
padded_width = (image_width + 31) - ((image_width + 31) % 32)


batch_size = (padded_height // 32) * (padded_width // 32)
height = 32
width = 32


padded_image = torch.zeros((1, 3, padded_height, padded_width))
padded_mask = torch.zeros((1, 1, padded_height, padded_width))
padded_image[0, :, :image_height, :image_width] = image
padded_mask[0, :, :image_height, :image_width] = mask


# print(padded_image.shape)
# print(padded_mask.shape)

with torch.no_grad():
    padded_image = Variable(padded_image)

import network

encoder = network.EncoderCell()
binarizer = network.Binarizer()
decoder = network.DecoderCell()

encoder.eval()
binarizer.eval()
decoder.eval()

encoder.load_state_dict(torch.load(args.model))
binarizer.load_state_dict(
    torch.load(args.model.replace('encoder', 'binarizer')))
decoder.load_state_dict(torch.load(args.model.replace('encoder', 'decoder')))

with torch.no_grad():
    encoder_h_1 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)),
                   Variable(torch.zeros(batch_size, 256, height // 4, width // 4)))

    encoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)),
                   Variable(torch.zeros(batch_size, 512, height // 8, width // 8)))

    encoder_h_3 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)),
                   Variable(torch.zeros(batch_size, 512, height // 16, width // 16)))

    decoder_h_1 = (Variable(torch.zeros(batch_size, 512, height // 16, width // 16)),
                   Variable(torch.zeros(batch_size, 512, height // 16, width // 16)))

    decoder_h_2 = (Variable(torch.zeros(batch_size, 512, height // 8, width // 8)),
                   Variable(torch.zeros(batch_size, 512, height // 8, width // 8)))

    decoder_h_3 = (Variable(torch.zeros(batch_size, 256, height // 4, width // 4)),
                   Variable(torch.zeros(batch_size, 256, height // 4, width // 4)))

    decoder_h_4 = (Variable(torch.zeros(batch_size, 128, height // 2, width // 2)),
                   Variable(torch.zeros(batch_size, 128, height // 2, width // 2)))

if args.cuda:
    encoder = encoder.cuda()
    binarizer = binarizer.cuda()
    decoder = decoder.cuda()

    with torch.no_grad():
        padded_image = padded_image.cuda()
        encoder_h_1 = (encoder_h_1[0].cuda(), encoder_h_1[1].cuda())
        encoder_h_2 = (encoder_h_2[0].cuda(), encoder_h_2[1].cuda())
        encoder_h_3 = (encoder_h_3[0].cuda(), encoder_h_3[1].cuda())

        decoder_h_1 = (decoder_h_1[0].cuda(), decoder_h_1[1].cuda())
        decoder_h_2 = (decoder_h_2[0].cuda(), decoder_h_2[1].cuda())
        decoder_h_3 = (decoder_h_3[0].cuda(), decoder_h_3[1].cuda())
        decoder_h_4 = (decoder_h_4[0].cuda(), decoder_h_4[1].cuda())




def image_to_patches(img, patch_size): 
    patches = []

    x_patches = img.shape[2] // patch_size
    y_patches = img.shape[3] // patch_size

    for i in range(x_patches):
        for j in range(y_patches):

            start_x = i * patch_size
            start_y = j * patch_size
            end_x = (i+1) * patch_size
            end_y = (j+1) * patch_size

            patch = img[0,:, start_x:end_x, start_y:end_y].unsqueeze(0)
            patches.append(patch)

    reshaped = torch.cat(patches, 0)
    return reshaped 


patches = image_to_patches(padded_image, 32)
mask_patches = image_to_patches(padded_mask, 32)

patch_quality = []
    
def get_quality2(mask_max):
    return (mask_max * 16) // 1 

def get_quality3(mask_max):
    return 8 + (mask_max * 8) // 1 

def get_quality4(mask_max):
    return (16 * (mask_max ** (1/3))) // 1
   
def get_quality5(mask_max):
    return (mask_max * 7) // 1 

def get_quality6(mask_max):
    return (16 * (mask_max ** (4))) // 1
   
def get_quality7(mask_max):
    if mask_max > 0.8:
        return 15
    else:
        return 0 
   

for mask_patch in mask_patches: 
    mask_max = mask_patch.max().item()

    
    if args.mapping == 1: 
        this_patch_quality = args.quality
    if args.mapping == 2: 
        this_patch_quality = get_quality2(mask_max)
    if args.mapping == 3: 
        this_patch_quality = get_quality3(mask_max)
    if args.mapping == 4: 
        this_patch_quality = get_quality4(mask_max)
    if args.mapping == 5: 
        this_patch_quality = get_quality5(mask_max)
    if args.mapping == 6: 
        this_patch_quality = get_quality6(mask_max)
    if args.mapping == 7: 
        this_patch_quality = get_quality7(mask_max)

    patch_quality.append(this_patch_quality)



# print(patch_quality)
# print ("patches.shape", patches.shape)
exit()

codes = []
for i in range(batch_size):
    codes.append([])

res = patches - 0.5


for iters in range(args.iterations):
    with torch.no_grad():

        # print ("res.shape", res.shape)

        encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
            res, encoder_h_1, encoder_h_2, encoder_h_3)

        code = binarizer(encoded)

        # print ("code.shape", code.shape)
        # print (code)
        output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4 = decoder(
            code, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4)

        res = res - output
        
        for i in range(batch_size):
            if patch_quality[i] >= iters:
                this_code = code[i,:,:,:].cpu().numpy()
                codes[i].append(this_code)

    print('Iter: {:02d}; Loss: {:.06f}'.format(iters, res.data.abs().mean()))

def save_codes(codes):

    packed_bits = [] 
    for i in range(batch_size):
        s = (np.stack(codes[i]).astype(np.int8) + 1) // 2
        export = np.packbits(np.array(s.reshape(-1)))
        packed_bits.append(export)

    if os.path.exists(args.output + ".npz"):
        os.remove(args.output + ".npz")

    np.savez_compressed(args.output, codes=packed_bits, width=image_width, height=image_height)



save_codes(codes)