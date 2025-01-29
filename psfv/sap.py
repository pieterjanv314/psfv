#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:44:02 2024

@author: Pieterjan Van Daele
"""
import os
import numpy as np

from psfv import acces_data


def get_raw_sap_lc(star_id,sector, mask_type='3x3',save_lc=True):
    '''
    Calculates a Simple Apereture Photometry (SAP) lightcurve. Only processing done is background substractions.
    if save_lc is True, then the lightcurve are saved in data/star_id/sector_xx/sap_{mask_type}.npy

    Parameters
    ----------
    star_id : string
        TESS identifier, of format 'TIC 12345678'
    sector : integer
        TESS sector, must be an non-zero integer
    mask_type : string, optional
        '1x1','3x3','5+' or '5x5. The latter mask consists of the central pixel and the up, down, left and right pixel.
    save_lc : boolean, optional
        If True, the SAP lightcurve fluxes are saved in data/star_id/sector_xx/sap_{mask_type}.npy

    Raises
    ------
    ValueError : If masl_type is not recognised.
        
    Returns
    -------
    times : 1D np.array 
        times of all cadences
    bk_flux : 1D np.array()
        local background flux per pixel in electrons/seconds
    '''
    times, bkg_flux = get_bk_lc(star_id,sector)
    tpf = acces_data.read_tpf(star_id,sector) 
    
    size = tpf.shape[1]
    c = size//2 #index indicating center

    #creating a aperture mask for SAP photometry
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
    elif mask_type == '5x5':
        for i in range(-2,3):
            for j in range(-2,3):
                mask[c+i][c+j] = True
    else:
        raise ValueError('Mask_type not recognised')
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
    Returns index that seperates the two orbits within a sector.

    Parameters
    ----------
    times : np.array()
        list of cadence times of only 1 sector.

    Raises
    ------
    ValueError : If the list of times range over more than 35 days.

    Returns
    -------
    index_half : integer
        index such that times[:index] belongs to orbit 1 of a sector and times[index:] to orbit 2.

    '''
    if 20>times[-1]-times[0]>35: #35 days, 1 sector should be 27-28 days
        raise ValueError('The list of cadence times seem not to correspond with a single TESS sector. They should span around 27-28 days.')

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
    Filters out slow lineair trends with lineair detrendning
    Substracts both half sectors seperatly with the best fitting straight line.
    The average 'flux' of the detrended lightcurve is thus close to zero. There is no rescaling or normalising of amplitude sizes.

    Parameters
    ----------
    lc_time ; np.array()
        list of cadence times
    lc_flux : np.array()
        list of fluxes or other observable that you want to detrend.

    Returns
    -------
    detrended_fluxes : np,array()
        list of lineair detrended fluxes.
    '''
    index_half = find_half_index(lc_time)
    lineair_fit1 = np.polyfit(lc_time[:index_half],lc_flux[:index_half],1)
    lineair_fit2 = np.polyfit(lc_time[index_half:],lc_flux[index_half:],1)
    detrended_fluxes= np.concatenate((lc_flux[:index_half]-(lineair_fit1[0]*lc_time[:index_half]+lineair_fit1[1]),lc_flux[index_half:]-(lineair_fit2[0]*lc_time[index_half:]+lineair_fit2[1])))

    return detrended_fluxes
