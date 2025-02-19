# -*- coding: utf-8 -*-
"""

Bayes Clumpy interpolation

Define innterpol() function to interpolate in values on a grid of  
clumpy torus models based on the CAT3D-WIND of
http://cat3d.sungrazer.org/

Se the paper of H\”onig, S. F., & Kishimoto, M. 2017, ApJ, 838, L20
for a description of parameters and models
https://iopscience.iop.org/article/10.3847/2041-8213/aa6838#apjlaa6838s2

Uses a trained Neural Network (ResNet of 32 layers, 128 neurons each, fully connected) which codifies 
the CAT3D-WIND model data and the resulting embedding was encoded using a CAE with a bottleneck of
32 elements.

Uses the hyperparameter file in .pth format (pyTorch) of both the NN and the CAE, together with the 
model.py containing the training model and model_encoder.py containing the model CAE. 

Autor: cwestend (https://github.com/cwestend/BClumpy_iNNterpol)

"""

import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.utils.data
import torch.nn as nn


import glob
import os
import sys

# Need scipy to work in np.longdouble format
#from scipy import linalg

# If linearly interpolating
# use https://github.com/cwestend/iNNterpol_PCA12/iNNterpol_PCA12.py

try:
    # hyperparam file 
    files = glob.glob("./*.pth")
    print('Reading the NN hyperparameter files: %s' % files[0])
    files_enc = glob.glob("./*.pth_encoder")
    print('Reading the CAE hyperparameter files: %s' % files_enc[0])
except:
    print('Cannot find the hyperparameter files')
    sys.exit()

def parse_input(in_values):
    try:
        #in_values = np.array(in_values, dtype=float32) 
        in_val = np.copy(np.array(in_values, dtype=float))
        if len(in_val) != 9:
            raise ValueError("Incorrect size of input parameters")
        else:
            # Logg in 10x format
            #in_val[4] = 10*in_val[4]
            return in_val
    except:
        str ="""Need an array-like with 9 params: a(index of power law), \n
                N0(umber of clouds along los), h(scale height, \n
                aw(radial distribution), theta_w(half-opening angle), theta_sig(angular width), \n
                f_wd(wind-to-disk ratio), rout(outer radius), ang(angle of los)
                """
        raise ValueError(str)

def innterpol(input_values):
    """ Function to interpolate in CAT3D-WIND data: input 10 values of
        a: index of the radial dust cloud distribution power law
        N0: number of clouds along an equatorial line-of-sight
        h: dimensionless scale height
        aw: radial distribution of dust clouds in the wind 
        theta_w: the half-opening angle of the wind 
        theta_sig: angular width of the wind 
        f_wd: wind-to-disk ratio 
        rout: outer radius
        tau_cl: (fixed = 50) optical depth of the individual clouds
        distrib: (fixed = 0) 
        ang: angle of line of vision
        (array like) and outputs an array(105) with (log)wavelength/freq variation of (log)flux F_nu (W/m^2) """

    try:
        # Import the model form local: there should be a model.py and a model_encoder.py in dir
        import model
        import model_encoder
    except:
        print('Cannot find the model files (model.py and model_encoder.py) in current directory')
        sys.exit()

    try:
        # Import the data lof x axis: log(frequency)/log(wavelength)
        data_file = np.load("./datos_clumpy_parm_x.npz", allow_pickle=True)
        print('Reading the data files for freq/wavelength: %s' % files[0])
        x_freq =  data_file['x_freq']
        x_wave  =  data_file['x_ldo']
    except:
        print('Cannot find the  data files for freq/wavelength')
        x_freq = []
        x_wave = []
        #sys.exit()



    inp_val = parse_input(input_values)

    # Rescale as trained (so they had similar size)
    par_a = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5]
    par_N0 = [5.0, 7.5, 10.0]
    par_h = [0.1, 0.2, 0.3, 0.4, 0.5]
    par_aw = [-2.5, -2.0, -1.5, -1.0, -0.5]
    par_theta_w = [30.0, 45.0]
    par_theta_sig = [7.0, 7.5, 10.0, 15.0]
    par_f_wd = [0.15, 0.3, 0.45, 0.6, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25]
    par_rout = [450.0, 500.0]
    # Inclination angles
    par_ang = [0.0,15.0,30.0,45.0,60.0,75.0,90.0]
    
    inp_val[0] = (inp_val[0]-min(par_a))/(max(par_a)-min(par_a))
    inp_val[1] = (inp_val[1]-min(par_N0))/(max(par_N0)-min(par_N0))
    inp_val[2] = (inp_val[2]-min(par_h))/(max(par_h)-min(par_h))
    inp_val[3] = (inp_val[3]-min(par_aw))/(max(par_aw)-min(par_aw))
    inp_val[4] = (inp_val[4]-min(par_theta_w))/(max(par_theta_w)-min(par_theta_w))
    inp_val[5] = (inp_val[5]-min(par_theta_sig))/(max(par_theta_sig)-min(par_theta_sig))
    inp_val[6] = (inp_val[6]-min(par_f_wd))/(max(par_f_wd)-min(par_f_wd))
    inp_val[7] = (inp_val[7]-min(par_rout))/(max(par_rout)-min(par_rout))
    inp_val[8] = (inp_val[8]-min(par_ang))/(max(par_ang)-min(par_ang))
    # tau_cl and distrib are fixed:
    inp_tau_cl = 1.
    inp_dist = 0.
    # Order of inputs when training
    inp_values = np.array((inp_val[0],inp_val[1],inp_val[2],inp_val[3],inp_val[4],inp_val[5],inp_val[6],
                          inp_val[7],inp_tau_cl,inp_dist,inp_val[8]))

    # Values for de-normalizing the flux
    # obtained for all models datos_clumpy_flat_flux(124740, 105) by:
    #max_tot = datos_clumpy_flat_flux.max(axis=1, keepdims=True)
    #min_tot = datos_clumpy_flat_flux.min(axis=1, keepdims=True)
    #min_n = round(min_tot.mean(),2)
    #max_n = round(max_tot.mean(),2)
    min_n = -8.32
    max_n = 5.66
    

    # The interpolating NN
    device = "cpu"
    checkpoint = max(files, key=os.path.getctime)
    chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    hyperparameters = chk['hyperparameters']
    model = model.Network(hyperparameters).to(device)
    model.load_state_dict(chk['state_dict'])
    model.eval()

    # The CAE that encodes/decodes (model with CAE weights for encoding/decoding)
    checkpoint_enc = max(files_enc, key=os.path.getctime)
    chk_enc = torch.load(checkpoint_enc, map_location=lambda storage, loc: storage)
    hyperparameters_enc = chk_enc['hyperparameters']
    model_encoder = model_encoder.Network(hyperparameters_enc).to(device)
    model_encoder.load_state_dict(chk_enc['state_dict'])
    model_encoder.eval()
    

    # Output of the NN
    # We only need the forward pass, so we do not accumulate gradients
    with torch.no_grad():
        global out_nn
        # We transform the input from Numpy to PyTorch tensor
        inputs = torch.tensor(inp_values.astype('float32')).to(device)
        out_nn = model(inputs)

    # Format to apply decoder (and recover the parameter stratification)
    out_nn_t = out_nn.unsqueeze(1).unsqueeze(0)

    # Applying the decoder on the predicted values by the NN
    # We only need the forward pass, so we do not accumulate gradients
    with torch.no_grad():
        global out_de
        inputs = out_nn_t.to(device)
        out_de = model_encoder.decoder(inputs)

    # Convert to array
    # We bring the result to the CPU (if it was on the GPU) and transform to Numpy
    out_de_n = out_de.cpu().numpy()

    # Transform to m,t,p and e parameters in depth
    inn_bclumpy_cae = np.empty((105),dtype=object)  

    inn_bclumpy_cae[:] = out_de_n[0,0,:]*(max_n-min_n)+min_n
   
    return inn_bclumpy_cae, x_freq, x_wave