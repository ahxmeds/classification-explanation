# classification-explanation

The work in this repository is associated with the following abstract:  
> S. Ahamed, C. F. Uribe, A. Rahmim, "Estimating the Tumor Localization Performance via Class-Activation Map Explanations of a Slice Classification Neural Network Without Pixel-Level Supervision." *EANM* 2023.

The work was submitted as an abstract to the European Association of Nuclear Medicine (EANM) 2023 congress held at Vienna, Austria from 9-13 September, 2023. The abstract can be found here: 
https://link.springer.com/article/10.1007/s00259-023-06333-x. 

This repository contains codes for the explanation of the classification model (with pretrained ResNet-18 backbone) which was fine-tuned on PET axial slices from a multi-centric lymphoma PET dataset (n=466).  

The explanations/interpretations of the classification model's predictions have been performed using four different methods, GradCAM, GradCAM++, EigenCAM, and ScoreCAM. We compute the centre-of-mass of these CAM-based heatmaps and generate circles of varying radii with this centre-of-mass as the centre to estimate a measure of the lesion localization on the axial slices of PET images. We also perform the Localization Receiver's Operating Characteristic (LROC) analysis and compare different methods based on area under the LROC curve.
