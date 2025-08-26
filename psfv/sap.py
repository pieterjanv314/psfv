#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:44:02 2024

@author: Pieterjan Van Daele

This module contains some standard light curve extraction techniques, including background estimation etc...
"""
import os
import numpy as np
import lightkurve as lk
from psfv import acces_data
import astropy.units as u


def get_raw_sap_lc(star_id,sector, mask_type='3x3',save_lc=True):
    '''
    Calculates a background substracted Simple Apereture Photometry (SAP) lightcurve.

    Parameters
    ----------
    star_id : string
        star identifier
    sector : integer
        TESS sector, must be an non-zero integer
    mask_type : string, optional
        '1x1','3x3','5+' or '5x5. The latter mask consists of the central pixel and the up, down, left and right pixel.
    save_lc : boolean, optional
        If True, the SAP lightcurve fluxes are saved in data/star_id/sector_xx/sap_{mask_type}.npy

    Raises
    ------
    ValueError : If mask_type is not recognised.
        
    Returns
    -------
    times : 1D np.array 
        times of all cadences
    bk_flux : 1D np.array()
        local background flux per pixel in electrons/seconds
    '''
    if os.path.isfile(f'data/{star_id}/sector_{sector}/'+f'sap_{mask_type}.npy'):
        background_corrected_flux = np.load(f'data/{star_id}/sector_{sector}/'+f'sap_{mask_type}.npy')
        times = np.load(f'data/{star_id}/sector_{sector}/'+f'times.npy')
    else:
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
        background_corrected_flux = target_lc.flux.value - bkg_mask_flux

        np.save(f'data/{star_id}/sector_{sector}/'+f'sap_{mask_type}.npy',background_corrected_flux)
        
    return times,background_corrected_flux

def get_bk_lc(star_id,sector):
    '''
    Calculates and saves a background lightcurve or reads an existing one if the file exists. 
    This light curve serves as an estimates for the time-dependent local background flux, which is calucalated as the average flux of all pixels without light sources.

    Parameters
    ----------
    star_id : string
        star identifier
    sector : integer
        TESS sector, must be an non-zero integer
        
    Returns
    -------
    times : 1D np.array 
        times of all cadences
    bk_flux : 1D np.array()
        local background flux in electrons/(second*pixel)
    '''
    file_path = f'data/{star_id}/sector_{sector}/'+'backgroundflux.npy'
    if os.path.exists(file_path):
        bk_flux = np.load(f'data/{star_id}/sector_{sector}/'+'backgroundflux.npy')
        times = np.load(f'data/{star_id}/sector_{sector}/'+'times.npy')
    else:     #let's create a background lightcurve
        tpf = acces_data.read_tpf(star_id,sector)

        background_mask = ~tpf.create_threshold_mask(threshold = 0.001, reference_pixel = None)
        nb_non_pixels = np.sum(np.nanmedian(tpf.flux.value,axis=0)<0.01)
        n_background_pixels = background_mask.sum()-nb_non_pixels
        background_lc_per_pixel = tpf.to_lightcurve(aperture_mask=background_mask) / n_background_pixels
        bk_flux = background_lc_per_pixel.flux.value
        np.save(f'data/{star_id}/sector_{sector}/'+'backgroundflux.npy',bk_flux)
        
        #save times of cadences, applicable for all lightkurves
        times = background_lc_per_pixel.time.value
        np.save(f'data/{star_id}/sector_{sector}/'+'times.npy',times)
    return times,bk_flux

def find_half_index(times):
    '''
    Returns index that seperates the two orbits within a sector. This is sometimes useful for detrending purposes.

    Parameters
    ----------
    times : np.array()
        list of cadence times of only 1 sector. in BTJD days

    Raises
    ------
    ValueError: If the list of times ranges over less then 24 days or more than 32 days, since one sector should be 27-28 days

    Returns
    -------
    index_half : integer
        index such that times[:index] belongs to orbit 1 of a sector and times[index:] to orbit 2.

    '''
    if 24>times[-1]-times[0]>32: #24-32 days, 1 sector should be 27-28 days
        raise ValueError('The list of cadence times seem not to correspond with a single TESS sector. They should span around 27-28 days.')

    dt = np.mean([times[i+1]-times[i] for i in range(10)])
    found = False
    i = len(times)//3
    while not found:
        if times[i+1]-times[i]>dt+0.05: #1.2h gap
            found = True
            index_half = i+1
        else:
            i += 1
    return index_half

def poly_detrending(lc_time, lc_flux, order=1,separate_halfsectors = False,return_polyval=False):
    '''
    Performs polynomial detrending.
    lc_flux is normalises with the best fitting polynomial of given order, the detrended flux is thus unitless and varies around 1.

    Parameters
    ----------
    lc_time : np.array()
        list of cadence times of on sector (only)
    lc_flux : np.array()
        list of fluxes that you wish to detrend, corresponding with lc_time
    order : int, optional
        order of the polynomial to fit (default is 1 for linear detrending)
    seperate_halfsectors : boolean, optional
        If True, the two orbits within the sector will be fitted and detrended seperatly.
    return_polyval : boolean, optional
        if True, returns additionally the 'y-values' of the fitted polenomial, sometimes useful for plotting purposes.


    Returns
    -------
    detrended_fluxes : np.array()
        list of polynomial detrended fluxes.
    '''

    if separate_halfsectors:
        index_half = find_half_index(lc_time)
        poly_fit1 = np.polyfit(lc_time[:index_half], lc_flux[:index_half], order)
        poly_fit2 = np.polyfit(lc_time[index_half:], lc_flux[index_half:], order)
    
        # Evaluate the polynomial fits
        poly_eval1 = np.polyval(poly_fit1, lc_time[:index_half])
        poly_eval2 = np.polyval(poly_fit2, lc_time[index_half:])
    
        detrended_fluxes = np.concatenate((lc_flux[:index_half]/poly_eval1, lc_flux[index_half:]/poly_eval2))
        poly_eval = np.concatenate((poly_eval1, poly_eval2))
    else:
        poly_fit = np.polyfit(lc_time, lc_flux, order)
        poly_eval = np.polyval(poly_fit, lc_time)
        detrended_fluxes = lc_flux/poly_eval
    
    if return_polyval:
        return detrended_fluxes,poly_eval
    else:
        return detrended_fluxes
    

def periodogram(time,flux):
    '''
    Converts the light curve to a Periodogram power spectrum. See the documentation of lightkurve.LightCurve.to_periodogram

    Parameters
    ----------
    time : np.array()
        list of cadence times of on sector (only)
    flux : np.array()
        list of fluxes corresponding with the time-array

    Returns
    -------
    freq : np.array()
        Array of frequencies
    power : np.array()
        Array of power-spectral-densities.
    '''
    lc = lk.LightCurve(time,flux)
    
    #calculate periodogram of light curve
    Nyquist_freq = 0.5*1/(lc.time[1]-lc.time[0])
    start = 1/(time[-1]-time[0])
    pg = lc.to_periodogram(minimum_frequency=start*1/u.day, maximum_frequency=Nyquist_freq, oversample_factor=20)
    freq = pg.frequency.value[:-1]
    power = pg.power.value[:-1]

    return freq,power