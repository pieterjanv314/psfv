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



    # with open('saved_dictionary.pkl', 'rb') as f:
    # loaded_dict = pickle.load(f)