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

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="exp.yaml", type=str)
parser.add_argument("--output", default="result_large_1.csv", type=str)
parser.add_argument("--min_width", default="100", type=int)
args = parser.parse_args()

with open(args.config, "rt") as fp:
    cfg = Namespace(**yaml.safe_load(fp))


def load_images():
    images = []

    dirs = os.listdir(cfg.data)
    for this_dir in dirs: 
        this_class = class_map[this_dir]
        imgs = os.listdir(cfg.data + this_dir)
        for img in imgs: 
            img_path = cfg.data + this_dir + "/" + img
            if "blurred" in img_path: 
                continue
            if "mask" in img_path: 
                continue
            if "compressed" in img_path: 
                continue
            if "decoded" in img_path: 
                continue

            images.append((img_path, this_class))

    random.shuffle(images)

    return images

    # return images


def generate_mask(img_path, mask_path, blurred_path):
    command = "python ../mask_generation/inference.py \
              --model {} --img_path {} --mask_path {} \
              --blurred_path {} \
              >/dev/null 2>&1".format(
                        cfg.mask_model, img_path, 
                        mask_path, blurred_path)

    os.system(command)

def compress_image(img_path, mask_path, code_path_noext, mapping, quality=0):
    command = "python encoder.py \
               --cuda --model {} --input {} \
               --mapping {} --mask {} --output {} --quality {}\
               >/dev/null 2>&1".format(
                            cfg.encoder_model, img_path,
                            mapping, mask_path, 
                            code_path_noext, quality)
    
    os.system(command)


def decode_image(code_path, decoded_path):
    command = "python decoder.py --cuda \
               --model {} --input {} --output {} \
               >/dev/null 2>&1".format(
                        cfg.decoder_model, 
                        code_path, 
                        decoded_path)

    os.system(command)


def classify_image(model_index, img_path, ground_truth):

    img = Image.open(img_path)
    img_t = transform_classifier(img)
    batch_t = torch.unsqueeze(img_t, 0)

    with torch.no_grad():
        model = classifier_models[model_index][1]
        out = model(batch_t)
        _, index = torch.max(out, 1)
        top5values, top5indices = torch.topk(out, 5)
        top5indices = top5indices[0].numpy()
        top5values = top5values[0].numpy()
        percentage = torch.nn.functional.softmax(out, dim=1)[0][ground_truth].item() * 100

    result = []

    for i in range(5):
        if ground_truth in top5indices[:i+1]:
            result.append(True)
        else:
            result.append(False)

    result.append(percentage)
    
    # print("%2.05f" % percentage[index[0]].item(), 
    #       "% is ", top5indices, 
    #       " ground truth: ",
    #       ground_truth, " ", class_map[ground_truth])
    
    return result
    

class_map = {}
class_name = {}
with open("loc.txt") as f:
    i = 0
    for line in f.readlines():
        s = line.strip().split(" ")
        class_map[s[0]] = i 
        class_map[i] = " ".join(s[1:]) 
        i+=1

classifier_models = [
    ("vgg11", models.vgg11(pretrained=True)),
    ("resnet_18", models.resnet18(pretrained=True)),
    ("resnet_34", models.resnet34(pretrained=True)),
    ("resnet_50", models.resnet50(pretrained=True)),
    ("resnet_152", models.resnet152(pretrained=True)),
    ("inception_v3", models.inception_v3(pretrained=True)),
]

for classifier in classifier_models:
    classifier[1].eval()

transform_classifier = transforms.Compose([    #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(224),                #[3]
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
    mean = [0.485, 0.456, 0.406],              #[6]
    std = [0.229, 0.224, 0.225]                #[7]
)])

def record_to_csv(file_path, name, img_class, setting, classifier, results, width, height):
    results_path = args.output

    with open(results_path, 'a+') as f:
        fieldnames = ["name", "image_class", "file_size", "width", "height", "setting", 
                      "classifier", "top1", "top2", "top3", 
                      "top4", "top5", "ground_truth_percentage", "file_path"]

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writerow({
            "name": name,
            "image_class": img_class, 
            "file_size": os.path.getsize(file_path),
            "width": width, 
            "height": height, 
            "setting": setting, 
            "classifier": classifier, 
            "top1": results[0],
            "top2": results[1],
            "top3": results[2],
            "top4": results[3],
            "top5": results[4],
            "ground_truth_percentage": results[5],
            "file_path": file_path,
        })

def classify_and_record(img_path, img_class, file_path, name, width, height, setting):
    for i in range(len(classifier_models)):
        classify_result = classify_image(i, img_path, img_class)
        record_to_csv(file_path, name, img_class, setting, classifier_models[i][0], classify_result, width, height)
        print(setting + ":", classify_result)

for img_path, img_class in load_images():
    try: 
        width, height = imagesize.get(img_path)

        if width > 1023 or height > 1023:
            continue
        if width < args.min_width:
            continue

        print("processing:", img_path)

        img_dentry = img_path.split("/")[-1]
        path = "/".join(img_path.split("/")[:-1])
        name = img_dentry.split(".")[0]
        ext = img_dentry.split(".")[1]

        mask_path = path + "/" + name + "_mask" + ".png"
        blurred_path = path + "/" + name + "_blurred" + ".jpg"

        generate_mask(img_path, mask_path, blurred_path)

        classify_and_record(img_path, img_class, img_path, name, width, height, "original")
        classify_and_record(blurred_path, img_class, blurred_path, name, width, height, "blurred")

        for encoder_mapping in range (1,8): 

            if encoder_mapping == 1:
                for quality in range (16):
                    suffix = str(encoder_mapping) + "_" + str(quality) 

                    code_path_noext = path + "/" + name + "_compressed_" + suffix
                    code_path = path + "/" + name + "_compressed_" + suffix + ".npz"
                    decoded_path = path + "/" + name + "_decoded_" + suffix + ".jpg"

                    compress_image(img_path, mask_path, code_path_noext, encoder_mapping, quality)
                    decode_image(code_path, decoded_path)
                    classify_and_record(decoded_path, img_class, code_path, name, width, height, "code_" + suffix)

            else:
                suffix = str(encoder_mapping)

                code_path_noext = path + "/" + name + "_compressed_" + suffix
                code_path = path + "/" + name + "_compressed_" + suffix + ".npz"
                decoded_path = path + "/" + name + "_decoded_" + suffix + ".jpg"

                compress_image(img_path, mask_path, code_path_noext, encoder_mapping)
                decode_image(code_path, decoded_path)
                classify_and_record(decoded_path, img_class, code_path, name, width, height, "code_" + suffix)

    except Exception as e:
        logging.error(traceback.format_exc())
