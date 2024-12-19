#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:41:44 2024

@author: c4072453
"""

import os
import pickle
import lightkurve as lk
import numpy as np
from astroquery.mast import Catalogs

# TODO: embed error if star_id is unrecognised. (or part of create_star_info)
# TODO: some sort of error if sector is not available for the requested star.
def download_tpf(star_id,sector=None):
    '''
    downloads and saves the following data:
        in data/star_id/sector_xx:
            TPF.fits: target-pixel-file file (i.e. the photometric images, see also Lightkurve documentation)
            flags.npy: TESS flag for each cadence
            
    TPFs have a default cutout size of 15x15 pixels.
    Overwrites any previous stored data.
    
    Parameters
    ----------
    star_id : string
        TESS identifier, of format 'TIC 12345678'
    sector : int, optional
        Option to specify a specific TESS sector. The default is the last available sector.

    Raises
    ------
    TypeError: If the sector is not an integer.
    ValueError: If the sector is not a positive integer.

    Returns
    -------
    None.

    '''
    if sector is not None:
        if not isinstance(sector, int):
            raise TypeError("Sector must be an integer.")
        if sector <= 0:
            raise ValueError("Sector must be a positive integer.")
    
    else: 
        star_info = read_star_info(star_id)
        sector = star_info['observed_sectors'][-1] # = the last available sector
    
    os.makedirs(f'data/{star_id}/sector_{sector}', exist_ok=True)
    filename = f'data/{star_id}/sector_{sector}/'+'TPF.fits'
    
    print(f'Downloading TPF of {star_id}, sector {sector}...')
    search_result = lk.search_tesscut(star_id, sector = sector)
    search_result.download(cutout_size=19).to_fits(output_fn = filename,overwrite = True)
    print('Download finished.')
    
    tpf = read_tpf(star_id,sector)
    
    #get a list a quality flags for each cadance
    np.save(f'data/{star_id}/sector_{sector}/'+'flags.npy',tpf.quality)

# TODO: embed error if star_id is unrecognised.   
# TODO: add RA and DEC
def create_star_info(star_id):
    '''
    returns a dictionary with the keys 'TIC_id', 'GAIA_id', 'Tmag_id', 'observed_sectors'.
    It saves or overwrites the dictionary in data/star_id/star_info.pkl
    
    You can read this file like this:
        
    with open('saved_dictionary.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    Parameters
    ----------
    star_id : string
        TESS identifier, of format 'TIC 12345678'
    '''
    #make a directory to save all the data
    os.makedirs(f'data/{star_id}', exist_ok=True)
    
    print('Searching target in online catalogues...')
    cat = Catalogs.query_object(star_id, catalog="TIC")[0]
    #cat.keys()
    
    #get list of available sectors
    search_result = lk.search_tesscut(star_id)
    print('Search finished.')
    sectors = [int(search_result.mission[i][-2:]) for i in range(len(search_result))]
    sectors.sort()
    
    #create dictionary
    star_info = {'TIC_id': star_id,
                 'GAIA_id': cat['GAIA'],
                 'Tmag_id': cat['Tmag'],
                 'observed_sectors': sectors}
    
    #save dictionary
    with open(f'data/{star_id}/star_info.pkl', 'wb') as f:
        pickle.dump(star_info, f)
    
    return star_info   

    
def read_star_info(star_id):
    '''
    reads data/star_id/star_info.pkl
    
    Parameters
    ----------
    star_id : string
        TESS identifier, of format 'TIC 12345678'

    Raises
    ------
    FileNotFoundError
        If the star_info.plk has not been created it yet. 
        It will the user if he wants to download it. If yes, create_star_info() is called

    Returns
    -------
    dict
        star_info dictionary, see create_star_info()

    '''
    try: 
        with open(f'data/{star_id}/star_info.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"The file 'data/{star_id}/star_info.pkl' does not exist. Can be created with create_star_info(star_id)")
        q = input('Do you want to do this now and continue: [y,n]: ')
        if q in {'y','Y','yes','Yes'}:
            return create_star_info(star_id)
        else:
            raise FileNotFoundError(f"The file 'data/{star_id}/star_info.pkl' does not exist.")
        
        
def read_tpf(star_id,sector):
    '''
    Reads a TPF.fits file and returns it as an targetpixelfile.TessTargetPixelFile object.
    
    Parameters
    ----------
    star_id : string
        TESS identifier, of format 'TIC 12345678'
    sector : int
        TESS sector. Must be >0

    '''
    try:
        tpf = lk.read(f'data/{star_id}/sector_{sector}/'+'TPF.fits')
        return tpf
    except FileNotFoundError:
        print(f"The file data/{star_id}/sector_{sector}/"+"TPF.fits' does not exist. The data of the requested star and sector must be dowlaoded first with download_tpf()")
        q = input('Do you want to do this now and continue: [y,n]: ')
        if q in {'y','Y','yes','Yes'}:
            return download_tpf(star_id,sector)
        else:
            raise FileNotFoundError(f"The file data/{star_id}/sector_{sector}/"+"TPF.fits' does not exist.")
                                        
def list_of_downloaded_sectors(star_id):
    '''
    Gives a list for which sectors, data has been dowloaded.
    
    Parameters
    ----------
    star_id : string
        TESS identifier, of format 'TIC 12345678'
        
    Returns
    -------
    sectors: list of integers
    '''
    sectors = []
    folder_path = f'data/{star_id}'
    for filename in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, filename)) and filename.startswith('sector_'):
            sector_number = filename[7:]  # Extract the number part from the filename
            if sector_number.isdigit() and len(sector_number) == 2:
                sectors.append(int(sector_number))
    return sectors

def tpf_roughqualitycheck_succesful(tpf):
    '''
    Does a quick and dirty quality check of the tpf, which fails if over 25% of the pixel fluxes are below 20 e/s.
    This usually indiciates that the target is either close to or over the edge of the CCD.
    
    Parameters
    ----------
    tpf: targetpixelfile.TessTargetPixelFile
    
    Returns
    -------
    Boolean :
        True if check is succesful, False if unseccesful.
    
    '''
    succes = True
    flux_values = tpf.flux.value.flatten()
    if len(flux_values[np.where(flux_values<20)])/len(flux_values)>0.26:
        # This means that many fluxes are very small. 
        # This usually indicates that the target is close to or over the edge of the CCD
        succes = False
    return succes