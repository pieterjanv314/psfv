#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:22:09 2024

@author: Pieterjan Van Daele
"""
from astropy.coordinates import SkyCoord
#import acces_data
import numpy as np
import matplotlib.pyplot as plt

from psfv import *


def quick_tpf_plot(tpf):
    '''
    Simple plot if median frame image for inspection purposes of TPF. The location of the target star is indicated in red.
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
    
def plot_background(star_id,sector):
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
        plt.scatter(flag_times, flag_bk, c='orange',s=1,label='Cadences with TESS flag')
        plt.legend()
    
    plt.xlabel('Time - 2457000 [BTJD days]',fontsize=8)
    plt.ylabel('Background flux (e/s)',fontsize=8)
    plt.show()
    
def check_fit_input_plot(fit_input,i_cad:int=234):
    #first we do a psf fit of a random frame
    tpf = acces_data.read_tpf(fit_input['star_id'],fit_input['sector'])
    bk_times,bk_fluxes = sap.get_bk_lc(fit_input['star_id'],fit_input['sector'])

    original_image = tpf.flux.value[i_cad]
    image = tpf.flux.value[i_cad]-bk_fluxes[i_cad]
    phot = fit_one_image(image,fit_input,print_result = True)

    #now let's make an inspection plot
    fig,ax = plt.subplots(1,2,figsize = (10,4))
    im_plt = ax[0].imshow(original_image,origin='lower',cmap = plt.cm.YlGnBu_r,alpha=0.4,norm='log',)
    im_plt = ax[0].imshow(give_central_cutout_image(original_image,new_length=cutoutsize), norm='log',origin = 'lower', cmap = plt.cm.YlGnBu_r,alpha=1)
    plt.colorbar(im_plt,ax=ax[0],label=r'$e^{-}/s$')

    ax[0].scatter(phot['x_init'].value,phot['y_init'].value,c='w',edgecolors='k',zorder=1,alpha=0.7) #gaia positions

    # Annotate each point with its index (plus one for 1-based indexing)
    for i, (x, y) in enumerate(zip(phot['x_init'].value,phot['y_init'].value)):
        ax[0].annotate(f'{i}', (x, y), textcoords="offset points", xytext=(0,5), ha='center',c='white')

    color='red'
    for k in range(len(phot)):
        fwhm, x, y = phot['fwhm_fit', 'x_fit', 'y_fit'][k]
        s = fwhm/2.355
        circle = plt.Circle((x, y), fwhm/2, color=color, lw=1.5,fill=False)
        ax[0].scatter(x,y,marker='+',color=color)
        ax[0].add_patch(circle)
        
    ax[0].tick_params(axis='x',which='both', bottom=False, top=False, labelbottom=False)
    ax[0].tick_params(axis='y',which='both', right=False, left=False, labelleft=False)
    ax[0].set_xlim(-1.5,19.5)
    ax[0].set_ylim(-1.5,19.5)
    ##################################################
    #residual image plot
    res_im = psfphot.make_residual_image(image)
    vmin = -np.max(np.percentile(image, 95))
    vmax = -vmin
    res_im_plt = ax[1].imshow(res_im, origin = 'lower', cmap = 'bwr',vmin=vmin,vcenter = 0, vmax=vmax, alpha=1)
    norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    plt.colorbar(res_im_plt,ax=ax[1],label = r'$e^{-}/s$')

    #draw a square to indicate the cutout
    rect = patches.Rectangle((len(res_im)//2 - cutoutsize // 2-0.5, len(res_im)//2 - cutoutsize // 2-0.5), cutoutsize, cutoutsize, linewidth=2, edgecolor='black', facecolor='none')
    ax[1].add_patch(rect)

    ax[1].tick_params(axis='x',which='both', bottom=False, top=False, labelbottom=False)
    ax[1].tick_params(axis='y',which='both', right=False, left=False, labelleft=False)
    ax[1].set_xlim(-1.5,19.5)
    ax[1].set_ylim(-1.5,19.5)

    plt.tight_layout()
    plt.show()