#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:22:43 2024

@author: Pieterjan Van Daele
"""
from psfv import acces_data
from psfv import psf_fit

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from astropy.table import Table

import signal
import pickle
import numpy as np

def _tic_handler(self,signum):
    print('the query of the TIC is taking a long time... Something may be wrong with the database right now...')

def query_TIC(target, target_coord, tic_id=None, search_radius=250.*u.arcsec, **kwargs):
        """
            Retrieving information from the TESS input catalog. 
            
            Parameters:
                target: target name
                target_coord (optional): target coordinates (astropy Skycoord)
                search_radius: TIC entries around the target coordinaes wihtin this radius are considered.
                **kwargs: dict; to be passed to astroquery.Catalogs.query_object or query_region.
        """
        
        deg_radius = float(search_radius / u.deg)
        
        tmag = None 
        nb_coords = []
        nb_tmags = []
        tic_index = -1
        
        try:
            # The TIC query should finish relatively fast, but has sometimes taken (a lot!) longer.
            # Setting a timer to warn the user if this is the case...
            signal.signal(signal.SIGALRM,_tic_handler)
            signal.alarm(30) # This should be finished after 30 seconds, but it may take longer...
            
            catalogTIC = Catalogs.query_region(target_coord, catalog="TIC", radius=deg_radius,**kwargs)
            ### NOTE: this catalogue also contains Gaia parameters. Relevant keywords include: 'GAIA', 'GAIAmag', 'e_GAIAmag'

            #print(catalogTIC.keys())
            signal.alarm(0)
            
        except:
            print(f"no entry could be retrieved from the TIC around {target}.")
            catalogTIC = []
        
        if(len(catalogTIC) == 0):
            print(f"no entry around {target} was found in the TIC within a {deg_radius:5.3f} degree radius.")
        
        else:
            if not (tic_id is None):
                tic_index = np.argmin((np.array(catalogTIC['ID'],dtype=int) - int(tic_id))**2.)
            else:
                tic_index = np.argmin(catalogTIC['dstArcSec'])
        
            if(tic_index < 0):
                print(f"the attempt to retrieve target {target} from the TIC failed.")
            
            else:
                ra = catalogTIC[tic_index]['ra']
                dec = catalogTIC[tic_index]['dec']
                tmag = catalogTIC[tic_index]['Tmag']
                
                # Retrieve the coordinates
                tess_coord = SkyCoord(ra, dec, unit = "deg")
                
                # Collecting the neighbours
                if(len(catalogTIC) > 1):
                    for itic, tic_entry in enumerate(catalogTIC):
                        if(itic != tic_index):
                            nb_coords.append(SkyCoord(tic_entry['ra'], tic_entry['dec'], unit = "deg"))
                            nb_tmags.append(tic_entry['Tmag'])
        
        nb_tmags = np.array(nb_tmags)
        
        return tmag, nb_coords, nb_tmags

#returns positions of stars to use as input guesses for the center of the PSFs
def get_pos(star_id,tpf, search_radius_pixels=5,max_tmag = 16,get_magnitudes = False):

    search_radius = search_radius_pixels*21*u.arcsec
    hdr = tpf.get_header()
    target_ra = hdr['RA_OBJ']
    target_dec = hdr['DEC_OBJ']

    target_coord = SkyCoord(target_ra, target_dec, unit = "deg")
    target_tmag, nb_coords, nb_tmags = query_TIC(star_id, target_coord,search_radius = search_radius)

    target_pixel = np.array([target_coord.to_pixel(tpf.wcs,origin=0)], dtype=float)
    sel_nb_pixels = np.array([nb_coord.to_pixel(tpf.wcs,origin=0) for nb_coord,nb_tmag in zip(nb_coords,nb_tmags) if (nb_tmag <= max_tmag)], dtype=float)
    sel_nb_tmags = nb_tmags[np.r_[nb_tmags <= max_tmag]]

    if len(sel_nb_pixels)>0:
        targetandnb_pixels = np.concatenate((target_pixel,sel_nb_pixels),axis=0)
        sel_tmags = np.concatenate((np.array([target_tmag]),sel_nb_tmags),axis=0)
    else:
        targetandnb_pixels = target_pixel
        sel_tmags = np.array([target_tmag])
    pos = Table(names=['x_0', 'y_0'], data=[targetandnb_pixels[:,0],targetandnb_pixels[:,1]])

    if get_magnitudes == True:
        return pos,sel_tmags
    else:
        return pos
    
def make_initial_guess_for_psf_fit(star_id,sector):
    if sector not in acces_data.list_of_downloaded_sectors(star_id):
        raise ValueError(f'data of sector {sector} is not available or has not been dowloaded yet. Try to download it first with acces_data.dowload_tpf()')
    
    tpf = acces_data.read_tpf(star_id,sector)

    
    # 1) learn which stars to include -->get_pos
    pos,Tmags = get_pos(tpf, search_radius_pixels=2,max_tmag = 15,get_magnitudes = True) #first element is the target position etc...

    # 2) for each included star, get gaia pos, estimate for sigma and integrated flux from Tmag
    initial_guesses = {}
    initial_guesses = pos.copy()
    initial_guesses['flux_0'] = 1.5e8*10**(-Tmags/2.5) #prefactor 1.5e8 is estimated from fits of single stars
    initial_guesses['sigma_0'] = 0.55



    # 3) build a python list and save it
    psf_results = {}
    psf_results['initial_guesses'] = initial_guesses
    psf_results['fit_results'] = 'not calculated yet'

    with open(f'data/{star_id}/sector_{sector}/psf_fit_results.pkl', 'wb') as f:
        pickle.dump(psf_results, f)
    return initial_guesses
        
    # with open('saved_dictionary.pkl', 'rb') as f:
    # loaded_dict = pickle.load(f)