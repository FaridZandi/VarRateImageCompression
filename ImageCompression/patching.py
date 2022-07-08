import torch
import torch.nn as nn
import math






torch.manual_seed(0)

patch_size = 8
channels = 4
height = 64 
width = 64 

img = torch.randint(10, (1, channels, height, width))
print (img.shape)
patches = image_to_patches(img, patch_size)
print (patches.shape)
reconstructed = patches_to_image(patches, patch_size, height, width)
print (reconstructed.shape)

print (img == reconstructed)
        
