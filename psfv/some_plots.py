#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 20:22:09 2024

@author: c4072453
"""
from astropy.coordinates import SkyCoord
#import acces_data
import numpy as np
import matplotlib.pyplot as plt


def quick_tpf_plot(star_id, tpf):
    
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
    

