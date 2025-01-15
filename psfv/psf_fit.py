#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:22:30 2024

@author: Pieterjan Van Daele

This file contains the script to calculate a PSF fit (of multiple stars combined) asusming the data is available and the initial conditions have been set. 
"""

from psfv import acces_data

import astropy
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
# from photutils.background import MADStdBackgroundRMS, MMMBackground
# from photutils.detection import IRAFStarFinder
from photutils.psf import (DAOGroup, IntegratedGaussianPRF, BasicPSFPhotometry)

from astropy.table import Table

import pickle
import numpy as np
import os

#object for PSF fitting, bkg_estimator = None
def create_photometry_object(sigma_fixed = False):
    daogroup = DAOGroup(crit_separation = 20)
    fitter = LevMarLSQFitter()
    psf_model = IntegratedGaussianPRF()
    psf_model.x_0.fixed = False
    psf_model.y_0.fixed = False

    if sigma_fixed == False:
        psf_model.sigma.fixed = False


    photometry = BasicPSFPhotometry(group_maker=daogroup,
                                    bkg_estimator=None, #I do my own background substraction to the image, because I do not understand what this one exactly does.
                                    psf_model=psf_model,
                                    fitter=LevMarLSQFitter(),
                                    fitshape=(11, 11))
    return photometry


#gives a square cutout. To use if you wanne give a smaller image to the PSF fit to make it quicker or to not let it get confused by bright sources further away.
def give_central_cutout_image(image,new_length=7):
    new_image = np.copy(image)
    n = len(image)
    start = n//2-new_length//2
    end = n//2+new_length//2
    for i in range(n):
        for j in range(n):
            if i<start or j<start or i>end or j>end:
                new_image[i][j] = np.nan
    return new_image


def get_result_tabs(star_id,sector,initial_guesses,overwrite=False,image_cutoutsize=7):
    psfresult_file = f'data/{star_id}/sector_{sector}/psf_fit_results.pkl'
    if os.exists(psfresult_file):
        with open(psfresult_file, 'rb') as f:  
            psf_results = pickle.load(f)
        if psf_results['fit_results'] != 'not calculated yet' and overwrite==False:
            return psf_results['fit_results']
        
    else:
        psf_results = {}
        psf_results['initial_guesses'] = initial_guesses
        psf_results['fit_results'] = 'not calculated yet'


        tpf = acces_data.read_tpf(star_id,sector)

        #now we start fitting
        all_result_tabs = []
        photometry = create_photometry_object()

        bk_times,bk_fluxes = acces_data.get_bk_lc(star_id,sector)
    
        print("Fitting all frames. Finished when the counter reaches 100")
        previous_precentage = -1
        for i in range(0,len(bk_times)):
            #let's keep track of how far we are.
            percentage = int(i/len(bk_times)*100)
            if percentage>previous_precentage:
                previous_precentage = percentage
                print(percentage,end=' ')
            
            image = tpf.flux.value[i]-bk_fluxes[i]  #image for photometry must be background subtracted, this is 2d-integer
            image = give_central_cutout_image(image,new_length = image_cutoutsize) #all values too far get NaN, only inner box is considered in fit
            if len(all_result_tabs) > 0:
                initial_guesses['x_0'] = all_result_tabs[-1]['x_fit']
                initial_guesses['y_0'] = all_result_tabs[-1]['y_fit']
                initial_guesses['sigma_0'] = all_result_tabs[-1]['sigma_fit']
                initial_guesses['flux_0'] = all_result_tabs[-1]['flux_fit']
            
            result_tab = photometry(image=image, init_guesses=initial_guesses)
            all_result_tabs.append(result_tab)

        psf_results['fit_results'] = np.array(all_result_tabs,dtype=Table)

        #save results
        with open(f'data/{star_id}/sector_{sector}/psf_fit_results.pkl', 'wb') as f:
            pickle.dump(psf_results, f)
        
        
        return psf_results

