# Evaluation-of-Image-to-Image-translation

Author: Yaoyang Zhang, Jianbo Chen, Yuting Ye


## Introduction
The goal of image-to-image translation is to learn a mapping between an input image and an output image. The problem can be further divided into several types. The first type has no training data and only one image is given for the algorithm. The objective is often defined clearly. For example, classical image denoising [4] is within this class. The second type contains training data with paired images. There are many works along this line [3, 8, 11, 10, 14]. The conditional GAN was used for this type of task by Isola et al. [10]. Isola et al. [10] also proposed a metric called FCN-score that measures the quality of image-to-image translation. The third type contains training data with unpaired images. In the latter two types, the objective itself is usually not defined clearly and needs to be learned. This leads to the difficulty of evaluating a concrete method on such tasks, together with the difficulty of evaluating the fundamental difficulty of such a task itself. In this report, we aim to tackle such problems for the third type of image-to-image translation.

## Overview
We proposed two metrics to evaluate the performance of CycleGAN on object transfer and style transfer. We defined the goodness of object transfer as the KL divergence between the class distribution of the original image and the "swapped" class distribution of the generated image calculated from a state-of-the-art ImageNet (in this case, we used the Inception V3 net) classification results. We then defined the goodness of style transfer as chi-square distance of the low-level features extracted from the first several layers of a state-of-the-art ImageNet.

We trained our own CycleGAN model and compare the results with those from the original paper based on our metric. Our metric is consistent with human perceptions and can be used for evaluating the performance of an image translation models (e.g. CycleGAN) on unpaired datasets.
