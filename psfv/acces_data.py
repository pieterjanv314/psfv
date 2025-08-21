#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:41:44 2024

@author: Pieterjan Van Daele

This module contains methods to download, store and access data.
"""

import os
import pickle
import lightkurve as lk
import numpy as np
from astroquery.mast import Catalogs
from astroquery.exceptions import RemoteServiceError

# TODO: embed error if star_id is unrecognised. (or part of create_star_info)
# TODO: some sort of error if sector is not available for the requested star.
def download_tpf(star_id,sector=None,coord=None,cutoutsize=19):
    '''
    downloads and saves the following data:
        in data/star_id/sector_xx:
            TPF.fits: target-pixel-file file (i.e. the photometric images, see also Lightkurve documentation)
            flags.npy: TESS flag for each cadence
            
    TPFs have a default cutout size of 19x19 pixels.
    Overwrites any previous stored data.
    
    Parameters
    ----------
    star_id : string
        TESS or GAIA identifier
    sector : int, optional
        Option to specify a specific TESS sector. The default is the last available sector.
    coord: astropy.coordinates.SkyCoord object, optional
        will be used when star_id is not recognised. The given star_id will still be used for naming any created files.
    cutoutsize: int, optional
        Defines the image size. Default is 19x19 pixels.
        an odd cutoutsize is recommended in order to have the target in the central pixel.
    Raises
    ------
    TypeError: If the sector is not an integer.
    ValueError: If the sector is not a positive integer.

    Returns
    -------
    None
    '''
    if sector is not None:
        if not isinstance(sector, int):
            raise TypeError("Sector must be an integer.")
        if sector <= 0:
            raise ValueError("Sector must be a positive integer.")
    
    else: 
        star_info = get_star_info(star_id)
        sector = star_info['observed_sectors'][-1] # = the last available sector
    
    os.makedirs(f'data/{star_id}/sector_{sector}', exist_ok=True)
    filename = f'data/{star_id}/sector_{sector}/'+'TPF.fits'
    
    try:
        print(f'Searching for {star_id}')
        search_result = lk.search_tesscut(star_id, sector = sector)
        print(f'Downloading TPF of {star_id}, sector {sector}...')
        search_result.download(cutout_size=cutoutsize).to_fits(output_fn = filename,overwrite = True)
        print('Download finished.')
    except Exception as e:
        if isinstance(e, RemoteServiceError):
            os.rmdir(f'data/{star_id}/sector_{sector}') #cleaning up what we started
            print('Sorry, looks like something is wrong with the online database at the moment. We got a RemoteServiceError:')
            print(f'{e}')
        else:
            print(f"An unexpected error occurred: {e}")
            print("Attempting to search using coordinates rather then identifier")
        
            if coord == None:
                raise ValueError('Star_id is not recognised by Lightkurve. Coordinates must be provided.')
            try:
                coord
                print('star_id not recognised, searching with coordinates instead.')
                search_result = lk.search_tesscut(coord,sector = sector)
                search_result.download(cutout_size=cutoutsize).to_fits(output_fn = filename,overwrite = True)

                tpf = read_tpf(star_id,sector,coord=coord)
                np.save(f'data/{star_id}/sector_{sector}/'+'flags.npy',tpf.quality)
            except:
                os.rmdir(f'data/{star_id}/sector_{sector}') #cleaning up what we started
                print('search failed')
        


def create_star_info(star_id,coord=None):
    '''
    Creates and returns a dictionary with the keys 'star_id', 'GAIA_id', 'Tmag','ra','dec', 'observed_sectors'.
    It saves or overwrites the dictionary in data/star_id/star_info.pkl
    
    You can acces this dictionary with :func:`~psfv.acces_data.get_star_info`
    
    Parameters
    ----------
    star_id : string
        TESS identifier, of format 'TIC 12345678'
    coord : astropy.coordinates.SkyCoord object
        searches with coordinates if the star_id is not recognised.

    Returns
    -------
    star_info: Python dictionary 
        a dictionary with the keys 'star_id', 'GAIA_id', 'Tmag','ra','dec', 'observed_sectors'
    '''
    #make a directory to save all the data
    os.makedirs(f'data/{star_id}', exist_ok=True)
    try:
        search_result = lk.search_tesscut(star_id) 
        if len(search_result) == 0:
                if coord == None:
                    raise ValueError('The query gave an empty result. This usually means that the star_id is not recognised by Mast.\nPerhaps try providing Coordinates.')
                else:
                    print('Attempting to search with coordinates instead')
                    search_result = lk.search_tesscut(coord)
                    cat = Catalogs.query_region(coord, catalog="TIC", radius=0.01)[0]
        else:
            cat = Catalogs.query_object(star_id, catalog="TIC")[0]

    except Exception as e:     
        if isinstance(e, RemoteServiceError):
            print('Sorry, looks like something is wrong with the online database at the moment. We got a RemoteServiceError:')
            print(f'{e}')
        else:
            print(f"An unexpected error occurred: {e}")



    #create dictionary
    star_info = {'star_id': star_id,
                'GAIA_id': cat['GAIA'],
                'Tmag': cat['Tmag'],
                'observed_sectors': sectors,
                'ra':cat['ra'],
                'dec': cat['dec']
                }

    sectors = [int(search_result.mission[i][-2:]) for i in range(len(search_result))]
    sectors.sort()
    #save dictionary
    with open(f'data/{star_id}/star_info.pkl', 'wb') as f:
        pickle.dump(star_info, f)
    
    return star_info   
    
def get_star_info(star_id:str,coord=None):
    '''
    reads and returns the dictionary stored in data/star_id/star_info.pkl. If that file does not exist, It is created by calling :func:`~psfv.acces_data.create_star_info`.
    
    Parameters
    ----------
    star_id : string
        TESS identifier, of format 'TIC 12345678'
    coord : astropy.coordinates.SkyCoord object
        Argument passed on to :func:`~psfv.acces_data.create_star_info` in case star_info does needs to be created first.

    Raises
    ------
    FileNotFoundError
        If the star_info.plk has not been created yet. 
        It will ask if you wants to download it. If yes, create_star_info() is called

    Returns
    -------
    dict
        star_info dictionary, see :func:`~psfv.acces_data.create_star_info`
    '''
    try: 
        with open(f'data/{star_id}/star_info.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
            star_info = create_star_info(star_id,coord=coord)
            return star_info

def read_tpf(star_id:str,sector:int,warn_ifnotdownloadedyet=False,coord=None):
    '''
    Reads a TPF.fits file and returns it as an targetpixelfile.TessTargetPixelFile object.
    
    Parameters
    ----------
    star_id : string
        TESS or GAIA identifier
    sector : int
        TESS sector. Must be >0
    warn_ifnotdownloadedyet : boolean
        Gives a warning if True and data/{star_id}/sector_{sector}/TPF.fits' does not exist yet. It requires user input to continue. 
        if False, :func:`~psfv.acces_data.download_tpf` will automatically be called, skipping the manual check.
    coord : astropy.coordinates.SkyCoord object
        searches with coordinates if the star_id is not recognised.

    Returns
    -------
    tpf: targetpixelfile.TessTargetPixelFile
        See the documentation of the Lightkurve python package

    '''
    try:
        tpf = lk.read(f'data/{star_id}/sector_{sector}/'+'TPF.fits')
        return tpf
    except FileNotFoundError:
        if warn_ifnotdownloadedyet:
            print(f"The file data/{star_id}/sector_{sector}/"+"TPF.fits' does not exist. The data of the requested star and sector must be downloaded first with download_tpf()")
            q = input('Do you want to do this now and continue: [y,n]: ')
            if q in {'y','Y','yes','Yes'}:
                download_tpf(star_id,sector,coord=coord)
                return lk.read(f'data/{star_id}/sector_{sector}/'+'TPF.fits')
            else:
                raise FileNotFoundError(f"The file data/{star_id}/sector_{sector}/"+"TPF.fits' does not exist.")
        else:
            download_tpf(star_id,sector,coord=coord)
            return lk.read(f'data/{star_id}/sector_{sector}/'+'TPF.fits')

def list_of_downloaded_sectors(star_id):
    '''
    Gives a list for which sectors a TPF has been downloaded, by checking if a folder 'star_id/sector_xx' exists.
    
    Parameters
    ----------
    star_id : string
        star identifier'
        
    Returns
    -------
    sectors: python list
        list of dowloaded sectors
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
    Does a quick quality check of the tpf, which fails if over 25% of the pixel fluxes are below 20 electrons/s.
    This usually indiciates that the target is either close to or over the edge of the CCD.
    
    Parameters
    ----------
    tpf : targetpixelfile.TessTargetPixelFile
        Can be accessed with :func:`~psfv.acces_data.read_tpf`
    
    Returns
    -------
    Boolean :
        True if check is succesful, False if unsuccesful.
    '''
    succes = True
    flux_values = tpf.flux.value.flatten()
    if len(flux_values[np.where(flux_values<20)])/len(flux_values)>0.26:
        # This means that many fluxes are very small. 
        # This usually indicates that the target is close to or over the edge of the CCD
        succes = False
    return succes