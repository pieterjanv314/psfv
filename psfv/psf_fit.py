#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:22:30 2024

@author: Pieterjan Van Daele

This file contains the script to calculate a PSF fit (of multiple stars combined) asusming the data is available and the initial conditions have been set. 
"""

from psfv import acces_data

import signal
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import QTable
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
from photutils.psf import *
import numpy as np

import warnings


def _tic_handler(self,signum):
    print('the query of the TIC is taking a long time... Something may be wrong with the database right now... \n Restarting Kernel&Run usually helps')

def _query_TIC(target, target_coord, tic_id=None, search_radius=250.*u.arcsec, **kwargs):
        """
            Retrieving information from the TESS input catalog. 
            
            Parameters:
                target: target name
                target_coord: target coordinates (astropy Skycoord)
                search_radius: TIC entries around the target coordinates wihtin this radius are considered.
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
            signal.alarm(20) # This should be finished after 20 seconds, but it may take longer...

            catalogTIC = Catalogs.query_region(target_coord, catalog="TIC", radius=deg_radius,**kwargs)
            ### NOTE: this catalogue also contains Gaia parameters. Relevant keywords include: 'GAIA', 'GAIAmag', 'e_GAIAmag'

            #print(catalogTIC.keys())
            signal.alarm(0)
            
        except:
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
                
                # Collecting the neighbours
                if(len(catalogTIC) > 1):
                    for itic, tic_entry in enumerate(catalogTIC):
                        if(itic != tic_index):
                            nb_coords.append(SkyCoord(tic_entry['ra'], tic_entry['dec'], unit = "deg"))
                            nb_tmags.append(tic_entry['Tmag'])
        
        nb_tmags = np.array(nb_tmags)
        
        return tmag, nb_coords, nb_tmags

#gives a square cutout. To use if you wanne give a smaller image to the PSF fit to make it quicker or to not let it get confused by bright sources further away.
def give_central_cutout_image(image,new_length:int=7):
    '''
    Returns a smaller square image with the same center as original image.
    Parameters
    ----------
    image : 2D-array
        A square image
    new_length : integer
        An odd integer defining the size of the new image.

    Returns
    -------
    new_image : 2D-array
        A smaller square image with size new_length
    '''

    if len(image) != len(image[0]):
        raise ValueError('Image must be a square NxN 2D array.')
    if new_length % 2 == 0:
        raise ValueError("n must be odd")    
    
    new_image = np.copy(image)
    n = len(image)
    start = n//2-new_length//2
    end = n//2+new_length//2
    for i in range(n):
        for j in range(n):
            if i<start or j<start or i>end or j>end:
                new_image[i][j] = np.nan
    return new_image

def create_mask(image,cutoutsize:int):
    '''
    Returns a mask with the same size as image that can be given 
    where mask == True, data is ignored for fit. 
    It includes data in a central square, simular to give_central_cutout_image(). There might be an updat in the future where we allow fore different mask shapes.

    Parameters
    ---------
    image : 2D-array
        A square image
    cutoutsize : integer
        an odd integer defining the size of the new image.

    Returns
    -------
    mask : 2D-array
        With same sizes as image. Where mask == True/False, data is ignored/included for fit
    '''
    if not cutoutsize % 2 != 0:
        raise ValueError('Cutoutsize must be an odd integer.')
    
    mask = np.full(image.shape, False, dtype=bool) #same shape as image but all elements are False
    n = len(image)
    start = n//2-cutoutsize//2
    end = n//2+cutoutsize//2
    for i in range(n):
        for j in range(n):
            if i<start or j<start or i>end or j>end:
                mask[i][j] = True
    return mask

#returns positions of stars to use as input guesses for the center of the PSFs
def get_pos(star_id,tpf, search_radius_pixels=5,max_tmag = 15,get_magnitudes = False):
    '''
    Searches for neighboring star on a TESS image.

    Parameters
    ----------
    star_id : string
        Target star identifier (e.g. GAIA DR3 xxxxxxxxxxxxxx)
    tpf: targetpixelfile.TessTargetPixelFile
        See also the documentation of the Lightkurve python package. Can be accesed with :func:`~psfv.acces_data.read_tpf`
    search_radius_pixels: float, optional
        Radius in which to look for neighbouring star in units of pixelsizes. Default is 5
    max_tmag : float, optional
        Star with TESS magnitudes above this value are excluded. Default is 15
    get_magnitudes : boolean, optional
        Whether to also return the TESS magnitudes of the found stars. Default is False
    
    Returns
    -------
    pos: python ictionary
        dictionary of positions of stars over the tpf image (in pixel units)
        pos['x_0'], a list x-positions, first element corresponds to the target star.
        pos['y_0'], a list x-positions, first element corresponds to the target star.
    sel_tmags: python list
        list of TESS magnitudes of stars, in the same order as 
        returned if get_magnitudes == True
    '''

    search_radius = search_radius_pixels*21*u.arcsec
    hdr = tpf.get_header()
    target_ra = hdr['RA_OBJ']
    target_dec = hdr['DEC_OBJ']

    target_coord = SkyCoord(target_ra, target_dec, unit = "deg")
    target_tmag, nb_coords, nb_tmags = _query_TIC(star_id, target_coord,search_radius = search_radius)

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

#object for PSF fitting, localbkg_estimator = None
def create_photometry_object(fwhm_fixed = False,fitshape:int=7):
    '''
    Creates a PSFPhotometry object (see Photutils package) to be used to perform psf photometry.

    Parameters
    ----------
    fwhm_fixed : boolean, optional
        if you want the fullwithhalfmaximum of the gaussian to be fixed to its initial condition. Default is False.
    fitshape : int, optional
        odd integer, defining the square box used around each single star for psf fit, see photutils documentation.

    Returns
    -------
    psfphot : PSFPhotometry
        A photutils.psf PSFPhotometry object.
    '''
    psf_model = CircularGaussianPRF()
    if fwhm_fixed == False:
        psf_model.fwhm.fixed = False
        #psf_model.sigma.fixed=False
    grouper = SourceGrouper(min_separation=10)          #for my purposes, let's put all the stars in 1 group by choosing min_sep large
    psfphot = PSFPhotometry(psf_model=psf_model,
                                grouper=grouper,        #see two lines above
                                fit_shape=fitshape,     #13 by default, this defines the included pixels in the fit: a 3x3 square around each included star.
                                finder = None, 
                                localbkg_estimator=None,#I do my own background substraction to the image, because I do not understand what this one exactly does.
                                fitter=LevMarLSQFitter(),#which numerical algorithm will be used
                                xy_bounds = 0.4,        #how far the positions are allowed to deviate from the initial condition, in pixel units
                                fitter_maxiters = 3000  #default was 100
                               )
    return psfphot    

def create_fit_input(star_id:str,
                sector:int,
                max_Tmag:float=15,#max Tess magnitude of neigbours included in the fit
                fitshape:int=3,        # size of square considered for fit of each star included
                radius_inculded:float=3, #for searching neighbours, in pixel units
                cutoutsize:int=15,      #mostly useless, but I'll leave it in, might be usefull later in some rare cases where the fit gets confused by something further out.
                delete_index=None    #sometimes, a visual check tells you than one item from the included stars should not be included, and it is just easiest to delete that one with the index. So it must be int or list of integers. Not to be used on a regular basis!
                ):
    '''
    Creates a dictionary containing all the manual input for the psf fit. Check whether this input makes sense with some_plots.check_fit_input_plot(). 
    Having a good working fit_input is key to all the future fits of this star/sector.

    Parameters
    ----------
    star_id : string
        target identifier (e.g. GAIA DR3 xxxxxxxxxxxxx, also works with TIC, ...)
    sector : integer
        TESS sector, must be an non-zero integer
    max_Tmag : float, optional
        Star with TESS magnitudes above this value are excluded. Default is 15
    fitshape : int, optional
        odd integer, defining the square box used around each single star for psf fit, see photutils documentation.
    radius_included: float, optional
        Radius in which to look for neighbouring star in units of pixelsizes. Default is 3
    cutoutsize : integer, optional
        an odd integer defining the size of the new image. Default is 15
    delete_index : int or list of integers
        Sometimes, a visual check tells you than one item from the included stars should not be included, and it is just easiest to delete that one using its index. Not to be used on a regular basis!
    
    Returns
    -------
    fit_input : python dictionary
        Dictionary containing all manual input for psf fits, to be used as argument for later functions.
    '''
    fit_input = {}
    fit_input['star_id'] = star_id
    fit_input['sector'] = sector
    fit_input['max_Tmag'] = max_Tmag
    fit_input['fitshape'] = fitshape
    fit_input['radius_included'] = radius_inculded
    fit_input['cutoutsize'] = cutoutsize
    fit_input['delete_index'] = delete_index
    return fit_input

def create_initial_parameters(fit_input:dict):
    '''
    Translates manual fit input (i.e. which stars to include) to stuff a computer can work with, i.e. initial conditions for the parameters of the psf fit

    Parameters
    ----------
    fit_input : dict
        Dictionary containing all manual input for psf fits. 
    
    Returns
    -------
    init_params : dict
        Initial parameters for psf fit
    '''
    tpf = acces_data.read_tpf(fit_input['star_id'], fit_input['sector'])
    pos,Tmags = get_pos(fit_input['star_id'],tpf, 
                        search_radius_pixels=fit_input['radius_included'],
                        max_tmag = fit_input['max_Tmag'],
                        get_magnitudes = True) #element 0 is the target position etc...
    if fit_input['delete_index']: #i.e. if not None
        print('Warning, an element of initial conditions gets deleted!')
        pos,Tmags = np.delete(pos,fit_input['delete_index']),np.delete(Tmags, fit_input['delete_index'])  
    
    #print(Tmags)
    init_params = QTable()
    init_params['x'] = np.array(pos['x_0'])
    init_params['y'] = np.array(pos['y_0'])
    init_params['flux'] = 1.5e8*10**(-np.array(Tmags)/2.5)

    return init_params

def print_photometry_results(phot):
    '''
    Prints an (ugly) overview of the psf phototmetry results. For quick inspection purposes only.

    Parameters
    ----------
    phot : QTable
        An astropy table with the PSF-fitting results (i.e. the output of :func:`~psfv.psf_fit.fit_one_image`)

    Returns
    -------
    None
    '''
    print('see photutils.psf documentation for more explanation.')
    print(phot['flux_init','flux_fit','flux_err'])
    print(phot['x_init','x_fit','x_err'])
    print(phot['y_init','y_fit','y_err'])
    print(phot['fwhm_init','fwhm_fit','fwhm_err'])
    #print(phot['local_bkg_init','local_bkg_fit','local_bkg_err'])

    d = np.sqrt((phot['x_init']-phot['x_fit'])**2+(phot['y_init']-phot['y_fit'])**2)
    print('distances from gaia position:',np.array(d))

    #print(psfphot.fit_info)
    print(phot['id','group_id','qfit','cfit'])
    print(phot['flags'])

def fit_one_image(image,init_params,fit_input,print_result = False,get_residual_image=False):
    '''
    Performs psf photometry on one single image. The image should already be background subtracted

    Parameters
    ----------
    image : 2D array
        2D array with fluxes of each pixel.
    init_params: dict
        initial conditions on fit parameters. see also :func:`~psfv.psf_fit.create_initial_params`.
    fit_input : dict
        Dictionary containing all manual input for psf fits.  see also :func:`~psfv.psf_fit.create_fit_input`.
    print_results : boolean,optional
        calls :func:`~psfv.psf_fit.print_photometry_results` if True, default is False.
    get_residual_image : boolean, optional
        Whether to return a residual image.

    Returns
    -------
    phot : QTable
        An astropy table with the PSF-fitting results (i.e. the output of fit_one_image)
    res_im : 2D array
        residual image, same size and unite as image parameter
        returned if get_residual_image = True    

    '''
    psfphot = create_photometry_object(fitshape=fit_input['fitshape'])

    phot = psfphot(image,init_params=init_params,mask=create_mask(image,cutoutsize=fit_input['cutoutsize']))
    if print_result:
        print_photometry_results(phot)
    if get_residual_image == True:
        return phot,psfphot.make_residual_image(image)
    else:
        return phot


