# TissueSegmentation

This reposotory contains the code to run a tissue-background segmentation based on [Resolution-agnostic tissue segmentation in whole-slide histopathology images with convolutional neural networks](https://doi.org/10.7717/peerj.8242). It was used in the acrobat challenge to mask away the control tissue, see [here](https://histoapp.pages.fraunhofer.de/about/2022-08-26-method_description_acrobat_challenge.pdf).

The software is provided on "AS IS" basis, i.e. it comes without any warranty, express or implied including (without limitations) any warranty of merchantability and warranty of fitness for a particular purpose.

## Requirements
The network was trained on two GPUs, but works on CPU for inference.
To call the inference code, you can build a docker image or a python package.

## Getting started

For a quick start, you can use the docker image. It outputs the biggest connected tissue segmentation, which was used to ignore the control tissue.
Go to `minimal_inference_docker` for more details on how to build and run the docker image.

For a more fine grained control, you can use the python package. It provides functionalities for tissue segmentation with and without post processing, which only chooses the biggest connected segmentation.
Go to `tissue_segmentation` for more details on how to build and run the python package.
