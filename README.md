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
## Data:

```

datos_clumpy_parm_x.npz - Small datafile with the values for the x-axis in frequency and wavelenght (as they are not evenly spaced for CAT3D-WIND data)
```

## Usage

Just run it in the directory with the above files to obtain the interpolated flux for the different parameters:  
a - index of the radial dust cloud distribution power law,   
N0 - number of clouds along an equatorial line-of-sight,  
h - dimensionless scale height,  
aw - radial distribution of dust clouds in the wind,  
theta_w - the half-opening angle of the wind,  
theta_sig - angular width of the wind,  
f_wd - wind-to-disk ratio, rout - outer radius,  
ang - angle of line of vision

```
% run "./BClumpy_innterpol.py.py"
% a = -2.5; N0 = 7.5; h = 0.2; aw = -1.5; theta_w = 45.; theta_sig = 15.; f_wd = 0.6; rout = 450.; ang = 60.
% input_val =  np.array((a, N0, h, aw, theta_w, theta_sig, f_wd, rout, ang))

% flux_innterp, x_freq, x_lam = innterpol(input_val)

```
