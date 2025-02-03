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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches

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
        plt.scatter(flag_times, flag_bk, c='orange',s=3,label='Cadences with TESS flag',zorder=1)
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
    
    init_params = psf_fit.create_initial_parameters(fit_input)
    psfphot_result,res_im = psf_fit.fit_one_image(image,init_params,fit_input,print_result = True,get_residual_image=True)

    #now let's make an inspection plot
    fig,ax = plt.subplots(1,2,figsize = (10,4))
    ax[0].set_title('TESS image')
    im_plt = ax[0].imshow(original_image,origin='lower',cmap = plt.cm.YlGnBu_r,alpha=0.4,norm='log',)
    im_plt = ax[0].imshow(psf_fit.give_central_cutout_image(original_image,new_length=fit_input['cutoutsize']), norm='log',origin = 'lower', cmap = plt.cm.YlGnBu_r,alpha=1)
    plt.colorbar(im_plt,ax=ax[0],label=r'$e^{-}/s$')

    ax[0].scatter(psfphot_result['x_init'].value,psfphot_result['y_init'].value,c='w',edgecolors='k',zorder=1,alpha=0.7) #gaia positions

    color='red'
    for k in range(len(psfphot_result)):
        fwhm, x, y = psfphot_result['fwhm_fit', 'x_fit', 'y_fit'][k]
        s = fwhm/2.355
        circle = plt.Circle((x, y), fwhm/2, color=color, lw=1.5,fill=False,label='FWHM')
        ax[0].scatter(x,y,marker='+',color=color)
        ax[0].add_patch(circle)
    
        # Annotate each point with its index (plus one for 1-based indexing)
    for i, (x, y) in enumerate(zip(psfphot_result['x_init'].value,psfphot_result['y_init'].value)):
        ax[0].annotate(f'{i}', (x, y), textcoords="offset points", xytext=(0,5), ha='center',c='magenta')

    ax[0].tick_params(axis='x',which='both', bottom=False, top=False, labelbottom=False)
    ax[0].tick_params(axis='y',which='both', right=False, left=False, labelleft=False)
    ax[0].set_xlim(-1.5,19.5)
    ax[0].set_ylim(-1.5,19.5)
    ax[0].legend()
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
    ax[1].set_xlim(-1.5,19.5)
    ax[1].set_ylim(-1.5,19.5)

    plt.tight_layout()
    plt.show()

def plot_psf_fitted_fluxes(psf_fit_results):
    star_id,sector = psf_fit_results['fit_input']['star_id'],psf_fit_results['fit_input']['sector']
    time,flux_sap = sap.get_raw_sap_lc(star_id, sector,mask_type='3x3')

    n_cad = len(psf_fit_results['fit_results'])
    n_stars = len(psf_fit_results['fit_results'][0]['flux_fit'])

    psf_fluxes = []
    for k in range(n_stars):
        psf_fluxes.append([psf_fit_results['fit_results'][i]['flux_fit'][k] for i in range(n_cad)])

    fig,ax = plt.subplots(n_stars+1,1)

    ax[0].plot(time,flux_sap,label='3x3 SAP target', lw=0.5, c='black')

    ax[1].plot(time,psf_fluxes[0],label=f'psf lc target',lw=0.5)
    for j in range(1,n_stars):
        ax[j+1].plot(time,psf_fluxes[j],label=f'psf lc nb {j}',lw=0.5)

    for i in range(len(ax)):
        ax[i].legend(fontsize=7)

    plt.tight_layout()
    plt.show()