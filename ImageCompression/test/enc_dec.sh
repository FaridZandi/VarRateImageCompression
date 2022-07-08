#!/bin/bash

image_path=test/images
code_path=test/codes
decoded_path=test/decoded
encoder_path=pretrained/encoder_epoch_00000001.pth
decoder_path=pretrained/decoder_epoch_00000001.pth

for i in {01..24..1}; do
  this_image=$image_path/kodim$i.png
  echo Encoding $this_image
  mkdir -p $code_path
  python encoder.py --model $encoder_path --input $this_image --cuda --output $code_path/kodim$i --iterations 16

  this_code=$code_path/kodim$i.npz
  echo Decoding $this_code
  mkdir -p $decoded_path/kodim$i
  python decoder.py --model $decoder_path --input $this_code --cuda --output $decoded_path/kodim$i
done
