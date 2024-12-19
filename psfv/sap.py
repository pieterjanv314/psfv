#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:44:02 2024

@author: c4072453
"""
import os
import numpy as np

from psfv import acces_data


def get_raw_sap_lc(star_id,sector, mask_type='3x3'):
    '''
    Mask must be either '1x1','3x3' or '5+'. The latter mask consists of the central pixel and the up, down, left and right picel.
    '''
    times, bkg_flux = get_bk_lc(star_id,sector)
    tpf = acces_data.read_tpf(star_id,sector) 
    
    size = tpf.shape[1]
    c = size//2 #index indicating center

    mask = [[False for i in range(size)] for i in range(size)]
    if mask_type == '1x1':
        mask[c][c] = True
    elif mask_type == '5+':
        mask[c][c] = True
        mask[c-1][c],mask[c+1][c],mask[c][c-1],mask[c][c+1] = True,True,True,True
    elif mask_type == '3x3':
        mask[c][c] = True
        mask[c-1][c],mask[c+1][c],mask[c][c-1],mask[c][c+1] = True,True,True,True
        mask[c-1][c-1],mask[c+1][c+1],mask[c-1][c+1],mask[c+1][c-1] = True,True,True,True
    else:
        raise ValueError('Mask_type not recognaised')
    mask=np.array(mask)
    
    target_lc = tpf.to_lightcurve(aperture_mask=mask)
    bkg_mask_flux = bkg_flux * np.sum(np.array(mask)) #bkg_flux is per pixel, multiply with the number of pixels in the mask.
    corrected_flux = target_lc.flux.value - bkg_mask_flux
    np.save(f'data/{star_id}/sector_{sector}/'+'sap_{mask_type}.npy',corrected_flux)
    
    return times,corrected_flux

    

def get_bk_lc(star_id,sector):
    '''
    Calculates and saves a background lightcurve or reads an existing one if the file exists. 
    This light curve serves as an estimates for the time-dependent local background flux, 
    which is calucalated as the average flux of all pixels without light sources.

    Parameters
    ----------
    star_id : string
        TIC id of target. format: 'TIC 12345678' .
    sector : integer
        TESS sector, must be an non-zero integer
        
    Returns
    -------
    times : 1D np.array 
        times of all cadences
    bk_flux : 1D np.array()
        local background flux per pixel in electrons/seconds
    '''
    file_path = f'data/{star_id}/sector_{sector}/'+'backgroundflux.npy'
    if os.path.exists(file_path):
        bk_flux = np.load(f'data/{star_id}/sector_{sector}/'+'backgroundflux.npy')
        times = np.load(f'data/{star_id}/sector_{sector}/'+'times.npy')
    else:     #let's create a background lightcurve
        tpf = acces_data.read_tpf(star_id,sector)

        background_mask = ~tpf.create_threshold_mask(threshold = 0.00001, reference_pixel = None)
        n_background_pixels = background_mask.sum()
        background_lc_per_pixel = tpf.to_lightcurve(aperture_mask=background_mask) / n_background_pixels
        bk_flux = background_lc_per_pixel.flux.value
        np.save(f'data/{star_id}/sector_{sector}/'+'backgroundflux.npy',bk_flux)
        
        #save times of cadences, applicable for all lightkurves
        times = background_lc_per_pixel.time.value
        np.save(f'data/{star_id}/sector_{sector}/'+'times.npy',times)
    return times,bk_flux

def find_half_index(times):
    '''
    Returns index such that times[:index] belongs to orbit 1 of a sector and times[index:] to orbit 2.
    This is sometimes usefull for detrending

    Parameters
    ----------
    times : np.array()
        list of cadence times.

    Returns
    -------
    integer.

    '''
    dt = times[1]-times[0]
    found = False
    i = len(times)//3
    while not found:
        if times[i+1]-times[i]>dt+0.1:
            found = True
            index_half = i+1
        else:
            i += 1
    return index_half



def lin_detrending(lc_time,lc_flux):
    '''
    normalises both half sectors seperatly with a straight line 
    '''
    index_half = find_half_index(times)
    lineair_fit1 = np.polyfit(times[:index_half],fluxes[:index_half],1)
    lineair_fit2 = np.polyfit(times[index_half:],fluxes[index_half:],1)
    fluxes= np.concatenate((fluxes[:index_half]-(lineair_fit1[0]*times[:index_half]+lineair_fit1[1]),fluxes[index_half:]-(lineair_fit2[0]*times[index_half:]+lineair_fit2[1])))
