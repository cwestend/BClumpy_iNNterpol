# BClumpy_iNNterpol
Neural Network interpolation over CAT3D-WIND AGN torus models for the new improved BayesClumpy 

We trained a new Neural Network over the more than 124k models of CAT3D-WIND (see http://cat3d.sungrazer.org/)
in the manner of https://github.com/cwestend/iNNterpol but adding a Residual Network (ResNet)
configuration to be able to cope with the deeper layer configuration.


The code is provided together with the trained weights of the CAE used and NN in order to be able to rapidly
reconstruct each model atmosphere. 


# Requirements:

```
Python 3.6/3.8 with Pytorch 11.1 (possibly 10.2 is ok)
```
## Files:

The model.py with the training model for the NN and the model_encoder.py for the CAE encoder/decoder used are needed aswell as the *.pth files with the hyperparameters resulting from the NN and CAE training.

```

model.py
model_encoder.py
<date-time>.pth
<date-time-2>.pth_encoder

```
