#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:22:43 2024

@author: Pieterjan Van Daele
"""
from psfv import acces_data
from psfv import psf_fit
from psfv import sap

import astropy.units as u
import numpy as np
from photutils.psf import CircularGaussianPRF
import pickle
import os



def extract_psf_flux(image, psf_result, n = 2,object_index = 0): #weighted mask
    #n is integer, how fine one pixel should be gridded (sorry for the bad explanation), inverse of size of 1 resolution element within one pixel for numerical integration

    x = psf_result['x_fit'].value[object_index]
    y = psf_result['y_fit'].value[object_index]
    s = psf_result['sigma_fit'].value[object_index]
        
    size = len(image) #image should always be a sqaure
    assert len(image) == len(image[0])
    total_flux = 0
        
    k = [-0.5+1/(2*n)] # fineness 
    stop = False
    while len(k) < n:
        k.append(k[-1]+1/n)

    #i,j label pixels; i horizontally (x), j vertically (y) in image
    for i in range(size):
        for j in range(size):
            if not np.isnan(image[i][j]):
                weight = 0
                    
                #l,k label subdevision of pixel for numerical integration
                for l in k:
                    for m in k:
                        weight += 1/n**2 * CircularGaussianPRF.evaluate(x=j+l,y=i+m,sigma=s,x_0=x,y_0=y,flux = 1) #we put flux to 1 so that the total weights will add up to 1
                
                total_flux += (weight*image[i][j])
    return total_flux

def get_psf_fit_results(fit_input:dict,overwrite=False,get_neighbour_lightcurves=False):
    star_id,sector = fit_input['star_id'],fit_input['sector']
    filename = f'data/{star_id}/sector_{sector}/psf_fit_results.pkl'
    if overwrite == False and os.path.isfile(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        tpf = acces_data.read_tpf(star_id,sector)
        bk_times,bk_fluxes = sap.get_bk_lc(star_id,sector)

        all_cadance_results = []
        target_lightcurve = []
        neighbour_lightcurves = []

        init_params = psf_fit.create_initual_parameters(fit_input)
        #loop over all cadances
        previous_precentage = -1
        for i_cad in range(len(tpf.flux.value)):
            #let's keep track of how far we are.
            percentage = int(i_cad/len(tpf.flux.value)*100)
            if percentage>previous_precentage+4:
                previous_precentage = percentage
                print(percentage,end=' ')

            #now, let's do the science
            image_with_background = tpf.flux.value[i_cad]
            image = image_with_background-bk_fluxes[i_cad] #2d - integer
            all_cadance_results.append(psf_fit.fit_one_image(image,init_params,fit_input))
            #

        psf_fit_results = {}
        psf_fit_results['fit_input'] = fit_input
        psf_fit_results['fit_results'] = all_cadance_results
        # Save the dictionary to a binary file
        with open(filename, 'wb') as f:
            pickle.dump(psf_fit_results, f)
        return psf_fit_results


def get_psf_lightcurve(star_id:str,sector:int,overwrite=False):
    raise NotImplementedError
    filename = f'data/{star_id}/sector_{sector}/psf_lcs.pkl'
    if overwrite == False and os.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        with open(f'data/{star_id}/sector_{sector}/psf_fit_results.pkl', 'rb') as f:
            psf_fit_results = pickle.load(f)

        all_lcs = {}
        all_lcs['fit_input'] = psf_fit_results['fit_input']


        tpf = acces_data.read_tpf(star_id,sector)
        bk_times,bk_fluxes = sap.get_bk_lc(star_id,sector)




    # with open('saved_dictionary.pkl', 'rb') as f:
    # loaded_dict = pickle.load(f)