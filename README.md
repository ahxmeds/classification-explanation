# classification-explanation

The work in this repository is associated with the following abstract:
```Estimating the Tumor Localization Performance via Class-Activation Map Explanations of a Slice Classification Neural Network Without Pixel-Level Supervision S. Ahamed, C. F. Uribe, A. Rahmim. 

The work was submitted as an abstract to the European Association of Nuclear Medicine (EANM) 2023 congress held at Vienna, Austria from 9-13 September, 2023. The abstract can be found here: 
https://link.springer.com/article/10.1007/s00259-023-06333-x

This repository contains codes for the explanation of the classification model (with pretrained ResNet-18 backbone) which was fine-tuned on PET axial slices from a multi-centric lymphoma PET dataset (n=466). The explanations/interpretations of the model's predictions have been performed using four different methods, GradCAM, EigenCAM, ScoreCAM, and LayerCAM. The CAM-based explanations (heatmaps) were thresholded to create binary mask which was used for performing the Localization Receiver's Operating Characteristic (LROC) analysis with respect to the ground truth masks.