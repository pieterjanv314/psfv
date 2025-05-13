#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:22:09 2024

@author: Pieterjan Van Daele
"""

from psfv import acces_data
from psfv import sap
from psfv import psf_fit

from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs
import astropy.units as u
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.ticker as ticker

def quick_tpf_plot(tpf):
    '''
    Simple inspection plot of median frame image for inspection purposes of TPF. The location of the target star is indicated in red.

    tpf: targetpixelfile.TessTargetPixelFile
        See also the documentation of the Lightkurve python package. Can be accesed with :func:`~psfv.acces_data.read_tpf`
    '''
    hdr = tpf.get_header()
    target_ra = hdr['RA_OBJ']
    target_dec = hdr['DEC_OBJ']
        
    # Querying the TIC for the target & its neighbours
    target_coord = SkyCoord(target_ra, target_dec, unit = "deg")

    target_pix = target_coord.to_pixel(tpf.wcs,origin=0)
    med_frame = np.nanmedian(tpf.flux.value,axis=0)
    
    im_mask = med_frame < 0.01
    masked_image = np.ma.masked_where(im_mask, med_frame)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=tpf.wcs)

    plt.imshow(np.log10(masked_image), origin = 'lower', cmap = plt.cm.YlGnBu_r, 
       vmax = np.percentile(np.log10(masked_image), 95),
       vmin = np.percentile(np.log10(masked_image), 5),alpha=1)
    
    ax.scatter(target_pix[0],target_pix[1],c='r')
    
    # Setting the axis limits for the plot
    size = len(tpf.flux[0][0])
    ax.set_xlim(0.5,size-0.5)
    ax.set_ylim(0.5,size-0.5)

    # Overlaying a fancy grid
    plt.grid(axis = 'both',color = 'white', ls = 'solid')
    plt.show()

def plot_background(star_id:str,sector:int):
    '''
    Plot the local background flux for a star during a specific sector. Data flaged by TESS is overplotted in orange.

    Parameters
    ----------
    star_id : string
        TESS identifier, of format 'TIC 12345678'
    sector : integer
        The TESS sector.
    '''
    try: 
        times = np.load(f'data/{star_id}/sector_{sector}/times.npy')
        background = np.load(f'data/{star_id}/sector_{sector}/backgroundflux.npy')
        flags =  np.load(f'data/{star_id}/sector_{sector}/flags.npy')
    except FileNotFoundError:
        raise FileNotFoundError('Run get_bk_lc() first to calculate the background flux!')
    
    flag_times = []
    flag_bk = []
    for i in range(len(flags)):
        if flags[i]!=0:
            flag_times.append(times[i])
            flag_bk.append(background[i])
    
    fig = plt.figure()
    
    plt.title(f'Local background flux: {star_id}, sector {sector}')
    plt.plot(times,background)
    if len(flag_bk)>0:
        plt.scatter(flag_times, flag_bk, c='orange',s=3,label='Cadences with TESS flag',zorder=1)
        plt.legend()
    
    plt.xlabel('Time - 2457000 [BTJD days]',fontsize=8)
    plt.ylabel('Background flux (e/s)',fontsize=8)
    plt.show()

def check_fit_input_plot(fit_input,i_cad:int=234,print_fit_result = True,save_fig=False):
    #first we do a psf fit of a random frame
    tpf = acces_data.read_tpf(fit_input['star_id'],fit_input['sector'])
    bk_times,bk_fluxes = sap.get_bk_lc(fit_input['star_id'],fit_input['sector'])

    original_image = tpf.flux.value[i_cad]
    image = tpf.flux.value[i_cad]-bk_fluxes[i_cad]
    
    init_params = psf_fit.create_initial_parameters(fit_input)
    psfphot_result,res_im = psf_fit.fit_one_image(image,init_params,fit_input,print_result = print_fit_result,get_residual_image=True)

    #now let's make an inspection plot
    fig,ax = plt.subplots(1,2,figsize = (10,4))
    ax[0].set_title('TESS image')
    im_plt = ax[0].imshow(original_image,origin='lower',cmap = plt.cm.YlGnBu_r,alpha=0.4,norm='log')
    im_plt = ax[0].imshow(psf_fit.give_central_cutout_image(original_image,new_length=fit_input['cutoutsize']), norm='log',origin = 'lower', cmap = plt.cm.YlGnBu_r,alpha=1)
    plt.colorbar(im_plt,ax=ax[0],label=r'$e^{-}/s$')

    ax[0].scatter(psfphot_result['x_init'].value,psfphot_result['y_init'].value,c='w',edgecolors='k',zorder=1,alpha=0.7) #gaia positions

    color='red'
    for k in range(len(psfphot_result)):
        fwhm, x, y = psfphot_result['fwhm_fit', 'x_fit', 'y_fit'][k]
        s = fwhm/2.355
        circle = plt.Circle((x, y), fwhm/2, color=color, lw=1.5,fill=False)
        ax[0].scatter(x,y,marker='+',color=color)
        ax[0].add_patch(circle)
    
        # Annotate each point with its index (plus one for 1-based indexing)
    for i, (x, y) in enumerate(zip(psfphot_result['x_init'].value,psfphot_result['y_init'].value)):
        ax[0].annotate(f'{i}', (x, y), textcoords="offset points", xytext=(0,5), ha='center',c='magenta')

    ax[0].tick_params(axis='x',which='both', bottom=False, top=False, labelbottom=False)
    ax[0].tick_params(axis='y',which='both', right=False, left=False, labelleft=False)
    ax[0].set_xlim(-0.5,18.5)
    ax[0].set_ylim(-0.5,18.5)
    ##################################################
    #residual image plot

    vmin = -np.max(np.percentile(image, 95))
    vmax = -vmin
    ax[1].set_title('residual image')
    res_im_plt = ax[1].imshow(res_im, origin = 'lower', cmap = 'bwr',vmin=vmin, vmax=vmax, alpha=1)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.colorbar(res_im_plt,ax=ax[1],label = r'$e^{-}/s$')

    #draw a square to indicate the cutout
    n = len(res_im)
    s = fit_input['cutoutsize']
    rect = patches.Rectangle((n//2 - s // 2-0.5, n//2 - s // 2-0.5), s, s, linewidth=2, edgecolor='black', facecolor='none')
    ax[1].add_patch(rect)

    ax[1].tick_params(axis='x',which='both', bottom=False, top=False, labelbottom=False)
    ax[1].tick_params(axis='y',which='both', right=False, left=False, labelleft=False)
    ax[1].set_xlim(-0.5,18.5)
    ax[1].set_ylim(-0.5,18.5)

    plt.suptitle(fit_input['star_id']+' s'+str(fit_input['sector']))
    plt.tight_layout()
    plt.show()
    if save_fig==True:
        fig.savefig(f'data/{fit_input['star_id']}/sector_{fit_input['sector']}/{fit_input['star_id']}_s{fit_input['sector']}_psf_plot.png')

def plot_psf_fitted_fluxes(psf_fit_results:dict,save_fig:bool=False):
    '''    
    Parameters
    ----------
    psf_fit_results : python dictionary
        dictionary containing the fitted parameters as well as initual conditions etc...
        This should be the output of :func:`~psfv.psf_lc.get_psf_fit_results`.
    save_fig : boolean, optional
        Default is False. If True, figure is saved in f'data/{star_id}/sector_{sector}/{star_id}_s{sector}_psf_fluxes.png'
    
    '''
    star_id,sector = psf_fit_results['fit_input']['star_id'],psf_fit_results['fit_input']['sector']
    time,flux_sap = sap.get_raw_sap_lc(star_id, sector,mask_type='3x3')
    n_cad = len(psf_fit_results['fit_results'])
    n_stars = len(psf_fit_results['fit_results'][0]['flux_fit'])

    psf_fluxes = []
    for k in range(n_stars):
        psf_fluxes.append([psf_fit_results['fit_results'][i]['flux_fit'][k] for i in range(n_cad)])

    fig,ax = plt.subplots(n_stars+1,1)
    plt.suptitle(star_id + f' s{sector} (& neighbours)',fontsize=8)

    ax[0].plot(time,flux_sap,label='3x3 SAP target', lw=0.5, c='black')

    ax[1].plot(time,psf_fluxes[0],label=f'psf lc target',lw=0.5)
    for j in range(1,n_stars):
        ax[j+1].plot(time,psf_fluxes[j],label=f'psf lc nb {j}',lw=0.5)

    for i in range(len(ax)):
        ax[i].legend(fontsize=7)
        ax[i].set_ylabel(r'flux ($e^-/s$)',fontsize=7)
    
    ax[-1].set_xlabel('Time - 2457000 [BTJD days]',fontsize=7)
    plt.tight_layout()
    plt.show()
    if save_fig==True:
        fig.savefig(f'data/{star_id}/sector_{sector}/{star_id}_s{sector}_psf_fluxes.png')

def scalesymbols(mags, min_mag, max_mag):
    """
        A simple routine to determine the scatter marker sizes, based on the TESS magnitudes. This is usefull for fancy tpf plots.
        
        Parameters:
            mags (numpy array of floats): the TESS magnitudes of the stars
            min_mag (float): the smallest magnitude to be considered for the scaling
            max_mag (float): the largest magnitude to be considered for the scaling
        Returns:
            sizes (numpy array of floats): the marker sizes
    """
    
    sizes = 60. * (1.1*max_mag - mags) / (1.1*max_mag - min_mag)
    
    return sizes

def fancy_tpf_plot(tpf,target_id='No target id specified',plot_grid=True,save_fig = False):
    '''
    Shows TPF pixel plot of median frame with GAIA positions of all stars below 17mag.
    
    Parameters
    ----------
    tpf: targetpixelfile.TessTargetPixelFile
        See also the documentation of the Lightkurve python package. Can be accesed with :func:`~psfv.acces_data.read_tpf`
    target_id : string, optional
        Only used to display error messages if any.
    plot_grid : boolean, optional
        wether to plot a dec ra grid, default is True
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=tpf.wcs)
    hdr = tpf.get_header()

    target_ra = hdr['RA_OBJ']
    target_dec = hdr['DEC_OBJ']
    target_coord = SkyCoord(target_ra, target_dec, unit = "deg")
            
    try:
        cat = Catalogs.query_object(star_id, catalog="TIC")[0]
    except:
        coord = SkyCoord(target_ra, target_dec, unit = "deg")
        cat = Catalogs.query_region(coord, catalog="TIC", radius=0.01)[0]

    target_tmag = cat['Tmag']
    
    if target_id != 'No target id specified':
        ax.set_title(target_id+f's{hdr['sector']}')
    
    max_plot_tmag = 17. # max. TESS magnitude of stars to be shown on the figure
    
    # Querying the TIC for the target & its neighbours
    tmag, nb_coords, nb_tmags = psf_fit._query_TIC(target_id, target_coord,search_radius=200.*u.arcsec)
    
    # select the median frame
    image = np.nanmedian(tpf.flux.value,axis=0)
    
    # Plotting the chosen frame
    # Plotting the image
    im_mask = image < 0.01
    masked_image = np.ma.masked_where(im_mask, image)
    
    implot = plt.imshow(np.log10(masked_image), origin = 'lower', cmap = plt.cm.YlGnBu_r, 
       vmax = np.percentile(np.log10(masked_image), 95),
       vmin = np.percentile(np.log10(masked_image), 5))
    
    #colorbar
    cbar = plt.colorbar(implot,label=r'$e^-/s$')
    # Define a function to format the ticks
    def format_tick(x,pos):
        return f"{int(10**x)}"
    # Set the tick formatter
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_tick))

    # Overlaying a fancy grid
    if plot_grid == True:
        plt.grid(axis = 'both',color = 'white', ls = 'solid')
    
    # Overplotting the (sufficiently bright) neighbouring stars
    sel_nb_pixels = np.array([nb_coord.to_pixel(tpf.wcs,origin=0) for nb_coord,nb_tmag in zip(nb_coords,nb_tmags) if(nb_tmag <= max_plot_tmag)], dtype=float)
    sel_nb_tmags = nb_tmags[np.r_[nb_tmags <= max_plot_tmag]]
    plt.scatter(sel_nb_pixels[:,0],sel_nb_pixels[:,1],s=scalesymbols(sel_nb_tmags,np.amin(sel_nb_tmags), np.amax(sel_nb_tmags)),c='w',edgecolors='k',zorder=1)
    
    # overplotting the selected star
    target_pix = target_coord.to_pixel(tpf.wcs,origin=0)
    ax.scatter(target_pix[0],target_pix[1],s=scalesymbols(target_tmag,np.amin(nb_tmags), np.amax(nb_tmags)),c='r',zorder=2,label=str(np.round(target_tmag,1)))
    
    # Setting the axis limits for the plot
    # Setting the axis limits for the plot
    size = len(tpf.flux[0][0])
    ax.set_xlim(0.5,size-0.5)
    ax.set_ylim(0.5,size-0.5)
    
    # Some custom commands to make a nice legend
    arr = np.full(1, 1)

    ax.scatter(99, 99, s=scalesymbols(8.*arr,np.amin(sel_nb_tmags), np.amax(sel_nb_tmags)), c='w', edgecolors='k', label='8')
    ax.scatter(99, 99, s=scalesymbols(12.*arr,np.amin(sel_nb_tmags), np.amax(sel_nb_tmags)), c='w', edgecolors='k', label='12')
    ax.scatter(99, 99, s=scalesymbols(16.*arr,np.amin(sel_nb_tmags), np.amax(sel_nb_tmags)), c='w', edgecolors='k', label='16')
    
    lgnd = ax.legend(title="TESS mag",loc="lower right",title_fontsize=7,fontsize=7)  

    plt.show()
    if save_fig==True:
        if target_id == 'No target id specified':
            raise ValueError('star id must be given in order to save the plot in the right directory.')
        fig.savefig(f'data/{target_id}/sector_{hdr['sector']}/{target_id}_s{hdr['sector']}_TPF_plot.png')

def plot_centroid_path(star_id:str,sector:int,skip_epochs:int=20,save_fig:bool = False):
    '''
    Provides a nice plot of the centroid path.

    Parameters
    ----------
    star_id : string
        TESS identifier, of format 'TIC 12345678'
    sector : integer
        The TESS sector.
    skip_epochs: integer
        increase to make the plot less crowded.
    save_fig : boolean
        Default False. If True, the plot is saved in f'data/{star_id}/sector_{sector}/{star_id}_s{sector}_centroid_path.png'
    '''
    filename = f'data/{star_id}/sector_{sector}/psf_fit_results.pkl'
    try:
        with open(filename, 'rb') as f:
            psf_fit_results = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f'{filename} not found. You must first run a PSF fit before the centroid path can be investigated')

    n = len(psf_fit_results['fit_results'])
    x0,y0 = psf_fit_results['fit_results'][0]['x_fit'][0],psf_fit_results['fit_results'][0]['y_fit'][0]

    x = [psf_fit_results['fit_results'][i]['x_fit'][0]-x0 for i in range(0,n,skip_epochs)]
    x_err = [psf_fit_results['fit_results'][i]['x_err'][0] for i in range(0,n,skip_epochs)]
    y = [psf_fit_results['fit_results'][i]['y_fit'][0]-y0 for i in range(0,n,skip_epochs)]
    y_err = [psf_fit_results['fit_results'][i]['y_err'][0] for i in range(0,n,skip_epochs)]

    times,bk_flux = sap.get_bk_lc(star_id, sector)

    times_plot = [times[i] for i in range(0,n,skip_epochs)]
    # %%
    fig, ax = plt.subplots()
    plt.title(f'{star_id}, s{sector} \n centroid path')
    for i in range(len(x)):
        # Create an ellipse patch.'
        ellipse = patches.Ellipse((x[i], y[i]), 2*x_err[i], 2*y_err[i],color='black', fill=False, alpha=0.02)
        
        # Add the ellipse patch to the axes
        ax.add_patch(ellipse)
    scat = ax.scatter(x,y,s=2,marker='.',c=times_plot, cmap='coolwarm')

    ax.set_aspect('equal')
    plt.colorbar(scat,label='Time - 2457000 [BTJD days]')
    plt.grid()
    plt.ylabel('pixel displacement')
    plt.xlabel('pixel displacement')
    plt.show()

    if save_fig==True:
        fig.savefig(f'data/{star_id}/sector_{sector}/{star_id}_s{sector}_centroid_path.png')
