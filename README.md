# VarRateImageCompression

Images usually have background parts that contain little semantic information. Common image compression techniques do not take that information into account when compressing the images. Storing less information for the background images can potentially lead to smaller image sizes while preserving the semantically rich parts of the image. In this paper, we present a modular approach for incorporating region-of-interest information in the image compression procedure, hoping to increase compression ratio for the images.

# Example 

The original image might look like this: 

![original image](https://github.com/FaridZandi/VarRateImageCompression/blob/master/original.JPEG)

We find the regions of interest for the image: 

![mask](https://github.com/FaridZandi/VarRateImageCompression/blob/master/mask.png)

Then, compress the background parts of the image more than the important parts: 

![recon](https://github.com/FaridZandi/VarRateImageCompression/blob/master/recon.jpg)

#Evaluations 

Our method could not beat the baseline models (compressing the whole image with fixed rates) in image classification task: 


![eval](https://github.com/FaridZandi/VarRateImageCompression/blob/master/ImageCompression/plots/ground_truth_percentage/ground_truth_percentage_resnet_152_annot.png)



