# WeedNet
No, not that kind of weed...
## Introduction
The model WeedNet classifies different weeds and crops. Different kinds of weeds have high impact on the yield of an acre or field. The most common way to fight them in an agricultural environment is to make heavy use of pesticides. A model that classifies different kinds of plants, that can occure on such an environment could be used to fight them in a ecologically safer way. 
## Current Status
So far this repository is still a WIP but feel free to contribute...

## Architecture
The model has three convolutional + normalization layers and 2 fully connected layers, with the last one having 12 out channels. When forwarding data through the model, after each convolution, max-pooling and delineatization is applied before data is passed through the fully connected layers eventually. Then a log-softmaxed tensor is returned.
