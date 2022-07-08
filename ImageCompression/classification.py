from torchvision import models
from torchvision import transforms
import torch

from PIL import Image

availble_models = dir(models)
print(availble_models)

model = models.resnet101(pretrained=True)

transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
)])


with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]


for i in range(24):

    path = "/u/faridzandi/csc2231/ImageCompression/test/images/kodim"
    path += "%02d" % (i + 1,)
    path += ".png"

    img = Image.open(path)
    img_t = transform(img)
    batch_t = torch.unsqueeze(img_t, 0)
    model.eval()
    out = model(batch_t)
    _, index = torch.max(out, 1)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100 
    print("%02d" % (i + 1,), 
          "%2.05f" % percentage[index[0]].item(), 
          "% is ", classes[index[0]])












