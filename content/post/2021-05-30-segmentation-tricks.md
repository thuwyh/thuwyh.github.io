---
categories:
- Deep Learning
comments: true
date: "2021-05-30T10:25:03Z"
header:
  teaser: /assets/segmentation/1.png
tags:
- Semantic Segmentation
- hyper column
title: Tricks of Semantic Segmentation
---

![HuBMAP - Hacking the Kidney](/assets/segmentation/4.png){: .align-center style="width:80%"}
HuBMAP - Hacking the Kidney
{: .align-caption style="text-align:center;font-size:smaller"}

Last month, I spent some time doing the “HuBMAP - Hacking the Kidney” competition on Kaggle. The goal of this competition is the implementation of a successful and robust glomeruli FTU detector. It is a classical binary semantic segmentation problem. This is my second semantic segmentation competition, and our team ended up in 43rd place and won a silver medal.

Although a mislabeled sample completely destroyed the public leaderboard and made competitors quite confused, it is still a good competition for illustrating tricks of semantic segmentation.

I would like to begin with the Lovasz-Hinge loss. This is from the paper “The Lovász Hinge: A Novel Convex Surrogate for Submodular Losses”, and it proved very powerful in many competitions. You can find a good implementation in the famous [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) library. Though the loss is powerful, it may be difficult to train in some cases. A work around is to combine the Lovasz-Hinge loss with some other loss, such as Dice loss or Cross Entropy loss. Some competitors may also use a two stage approach: first train the model with Dice loss and then switch to the Lovasz-Hinge loss.

As for the model part, almost all competitors used Unet and its variants. The winner said he used the hypercolumn and attention mechanism with classical Unet in his [solution post](https://www.kaggle.com/c/hubmap-kidney-segmentation/discussion/238198). I also tried these two tricks, however, they did not improve my results significantly. As shown below, hypercolumn is very straightforward. It concatenates the resized feature maps from different layers to enhance the model capacity.


![Hypercolumn](/assets/segmentation/1.png){: .align-center style="width:80%"}
Hypercolumn
{: .align-caption style="text-align:center;font-size:smaller"}

Attention Unet is from the paper “Attention U-Net:Learning Where to Look for the Pancreas”. As shown below, the authors added attention gates(AGs) to the decoder part of the Unet. Commonly used attention mechanism in Kaggle competitions is a little bit different from the original paper. You can refer to either the winner’s post and the segmentation_models.pytorch repo. 

![Attention U-Net](/assets/segmentation/2.png){: .align-center style="width:80%"}
Attention U-Net
{: .align-caption style="text-align:center;font-size:smaller"}

By the way, there are several common attention structures, such as SCSE and CBAM. The famous competitor Bestfitting said he prefered CBAM. You can find some more detailed instructions here and here.

I also tried deep supervision in this competition. Basically speaking, deep supervision calculates loss at different scales. It is easy to implement, while it did not provide much performance gain to me.

In this competition, the original input images are very large. Competitors need to first split the original image into small patches and then train and inference the model. To avoid information loss, the sliding step is smaller than the patch size, so that there is some overlap area. However, we found that the model is still weaker for the boundary area. Thus, we used a weighted approach to reconstruct the output. As shown below, If a pixel was predicted several times(green area), we gave smaller weight to the predictions when this pixel is in the boundary area(patch 2) and larger weight to those in the inner area(patch 1). This trick gave us a nice boost.

![Example of our reconstruction strategy](/assets/segmentation/3.png){: .align-center style="width:80%"}
Example of our reconstruction strategy
{: .align-caption style="text-align:center;font-size:smaller"}


Training semantic segmentations requires lots of computing. I mainly used U-Net with EfficientNet as the encoder. With my current 24GB RTX 6,000 GPU in the HP Z4 Workstation, I could train the models with 640 x 640 images and a batch size of 16. My training set consisted of about 18,000 images, and an epoch took me only 17 or 18 minutes. If you lack computing power, you may also try the sampling tricks. Since the negative samples are much more than positive ones, you can random sampling some negative samples, instead of using all of them. This could not only reduce the time for converging but also save a lot of training time.

Last but not least, always trust your local cross validation. Some competitors tried to overfit the mislabeled public test sample and got poor results in the end. We only tried to improve our CV score, and we finally jumped up more than 100 places.  Unfortunately, we missed the submission with the highest private score. Otherwise, we would have won a gold medal :P



