#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:22:43 2024

@author: Pieterjan Van Daele
"""
from psfv import acces_data
from psfv import psf_fit
from psfv import sap

import numpy as np
from photutils.psf import CircularGaussianPRF
import pickle
import os


def extract_psf_flux(image, psf_result, n = 2,object_index = 0): #weighted mask
    '''
    This is an old version, to be updated if relevant.
    '''
    raise NotImplementedError
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
def read_psf_fit_results(star_id,sector):
    '''
    Reads previously calculated and saved psf fit results.

    Parameters
    ----------
    star_id : string
        TESS identifier, of format 'TIC 12345678'
    sector : int
        TESS sector. Must be >0
                
    Returns
    -------
    psf_fit_results : python dictionary
        dictionary containing the fitted parameters as well as initual conditions etc...
    '''
    filename = f'data/{star_id}/sector_{sector}/psf_fit_results.pkl'
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('No previously calculated result is available. Perform the psf fit with psf_lc.get_psf_fit_results()')

def get_psf_fit_results(fit_input:dict,overwrite=False):
    '''
    Performs PSF photometry on every cadance of a sector (a step towards building a psf lightcurve). Results are saved in data/{star_id}/sector_{sector}/psf_fit_results.pkl.
    It reads and returns previous stored results, unles overwrite is set True.
    Prints percentages 0 5 10 15 ... to keep track how far we got (expected to take a couple minutes on a normal pc)
    
    Parameters
    ----------
    fit_input : python dictionary
        to be create with :func:`~psfv.psf_fit.create_fit_input`. This parameter is a textbook example of 'garbage in, garbage out', so make sure to check if your fit_input makes sense with :func:`~psfv.some_plots.check_fit_input_plot`.
    overwrite : boolean, optional
        Overwrites previous stored results if True. Default is False (i.e. it just reads and returns a previous stored results if that exists).
    
    Returns
    -------
    psf_fit_results : python dictionary
        dictionary containing the fitted parameters as well as initual conditions etc...
        This is also saved in data/{star_id}/sector_{sector}/psf_fit_results.pkl
    '''
    star_id,sector = fit_input['star_id'],fit_input['sector']
    filename = f'data/{star_id}/sector_{sector}/psf_fit_results.pkl'

    if overwrite == False and os.path.isfile(filename):
        with open(filename, 'rb') as f:
            stored_result = pickle.load(f)
        if stored_result['fit_input'] != fit_input:
            print(f'WARNING: This is a previously stored psf fit result for {star_id}, sector {sector} but with different fit_input!\n Choose overwrite=True to recalculate the psf with your fit_input')
        return stored_result
    else:
        tpf = acces_data.read_tpf(star_id,sector)
        bk_times,bk_fluxes = sap.get_bk_lc(star_id,sector)

        all_cadance_results = []

        init_params = psf_fit.create_initial_parameters(fit_input)
        #loop over all cadances
        previous_precentage = 0
        print('this might take a couple minutes... Feel free to grab a coffee.\nThe counter below displays every 5% step reached.')
        for i_cad in range(len(tpf.flux.value)):
            #let's keep track of how far we are.
            percentage = int(i_cad/len(tpf.flux.value)*100)
            if percentage>=previous_precentage+5:
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
    '''
    to be implemented if the flux parameter of the fit results proof to be of insufficient quality. Then I'm gonna do a weighted sum over the pixels. 
    '''

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