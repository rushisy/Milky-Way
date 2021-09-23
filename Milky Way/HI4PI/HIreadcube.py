#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 10:48:55 2021

Read the FITS cube data of the HI4PI survey and generate an RGB image.

Data can be obtained from: https://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=J/A%2bA/594/A116/allsky_gal then "Submit",
then download the CAR.fits (Cartesian coordinates). This file is 33 GB, so download where you have fast internet.

@author: trungha
"""

#%% Initiation, only need to be run once 

import os
# os.chdir("/mnt/RESEARCH_DOCS/VSF_Turbulence")
os.chdir("F:\\OneDrive - UNT System\\RESEARCH_DOCS\\VSF_Turbulence")
import numpy as np
import PIL

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord

from tqdm import tqdm
import matplotlib as mpl
mpl.rcParams['font.family'] = 'stixgeneral'
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Edit these lines to link to the fits file with data
if os.name == 'nt':
    fits_data = fits.open('F:\\Gemini_QSO-main\\CAR.fits')
else:
    fits_data = fits.open('/mnt/RESEARCH_DOCS/VSF_Turbulence/HIsurvey/CAR_A01.fits')
    
# get_cubeinfo is used to translate from fits header to galactic coordinates in the image
from get_cubeinfo import get_cubeinfo    

HI_data = get_cubeinfo(fits_data[0].header) # coordinate info are stored here
hi4pi = fits_data[0].data # flux info in x-y-v array are stored here

#%% Some specifications of the region we want to plot

# Save some common molecular clouds for quick lookup
cloud_dict = {
    'All':{
        'range':np.array([0,360,-90,90]), #galactic longitude and latitude range
        'distance':200 #distance of the cloud to Earth
        },
    'LMC':{
        'range':np.array([270,310,-55,-30]),
        'distance':50e3
        },
    'Orion':{
        'range':np.array([190,218,-27,-7]),
        'distance':412
        },
    'Perseus':{
        'range':np.array([157,167,-20,-10]),
        'distance':320
        },
    'Taurus':{
        'range':np.array([165,177,-23,-12]),
        'distance':140
        },
    'Ophiuchus':{
        'range':np.array([345,13,9.5,30]),
        'distance':140
        }
    }

cloud = 'Perseus' # optional, input "All" if you want the entire map generated. If this is left blank, line 84 can be modified
# for the exact coordinate range you want to plot

#initiate a spatial boundary within which the flux is plotted
if list(cloud_dict.keys()).__contains__(cloud): 
    physical_box = cloud_dict[cloud]['range']
else:
    physical_box = np.array([181,201,-55,-35]) # first two numbers is longitude range, last two numbers is lattitude range.
if physical_box[1] <= physical_box[0]:
    physical_box[1] = physical_box[1] + 360
    
#%% Create velocity-dependent rgb map

velo_bins = HI_data[2] #array of velocity along the 3rd dimension of the datacube

gl_min, gl_max = physical_box[:2]
gb_min, gb_max = physical_box[2:]

# Read in data and trim to designated region 
gl, gb, vlsr, header_arrs = get_cubeinfo(fits_data[0].header, origin=1, returnHeader=True)
hi4pi_wcs = WCS(header_arrs[0])
all_gl = gl.flatten()
all_gb = gb.flatten()

patch_ind = np.all([all_gl>gl_min, all_gl<gl_max, 
                    all_gb>gb_min, all_gb<gb_max], axis=0)
if gl_max > 360:
    patch_ind = np.all([all_gl>gl_min, all_gl<360, 
                    all_gb>gb_min, all_gb<gb_max], axis=0)
    patch_ind_2 = np.all([all_gl>0, all_gl<gl_max-360, 
                    all_gb>gb_min, all_gb<gb_max], axis=0)
    patch_ind = patch_ind | patch_ind_2
patch_gl = all_gl[patch_ind]
patch_gb = all_gb[patch_ind]

# The next 32 lines is a parallelized 'for' loop to scan through the entire frame and sum up the flux within each predefined color
# channel. For this code, blue channel is defined from -100 km/s to -10 km/s (blueshifted toward Earth), green channel is between
# -10 km/s and 10 km/s (neutral), and red channel from 10 km/s to 100 km/s (redshifted away from Earth).
def generate_rgb(m, data, patch_gl, patch_gb, blocks):
    """
    m:          the number of individual pixels, imported from patch_gl on line 109
    data:       full array of flux
    patch_gl:   the list of galactic longitudes
    patch_gb:   the list of galactic latitudes
    blocks:     position of velocity indexes (e.g.: [-100,-10,10,100] km/s, then look at velo_bins to convert to index)
    """
    i = int(m)
    px, py = hi4pi_wcs.all_world2pix(patch_gl[i], patch_gb[i], 0)
   
    rgb_block[int(py),int(px),0] = np.nansum(data[blocks[2]:blocks[3],int(py),int(px)]) # vlsr 10 to 100 km/s
    rgb_block[int(py),int(px),1] = np.nansum(data[blocks[1]:blocks[2],int(py),int(px)]) # vlsr -10 to 10 km/s
    rgb_block[int(py),int(px),2] = np.nansum(data[blocks[0]:blocks[1],int(py),int(px)]) # vlsr -100 to -10 km/s
    return rgb_block[int(py),int(px),:]

m = list(range(patch_gl.size))
rgb_block = np.zeros((gl.shape[0],gl.shape[1],3))

from functools import partial
import multiprocessing
def parallel_runs (m):
    a_pool = multiprocessing.Pool(processes=8) # Change the number after 'process = ' to indicate number of threads to run on
    prod_m = partial(generate_rgb, data=hi4pi,patch_gl=patch_gl,patch_gb=patch_gb,blocks=[315,465,477,626])
    result_parallel = a_pool.map(prod_m,m)
    return result_parallel

# This returns rgb_block, which is raw flux sum for each pixel. This data still need to be converted to color
if __name__ == '__main__':
    mapout = parallel_runs(m)
   
# If your computer can't run in parallel, comment the 32 lines above and uncomment the next 6 lines. NOTE: this is very slow
# for i in tqdm(range(patch_gl.size)):
#    px, py = hi4pi_wcs.all_world2pix(patch_gl[i], patch_gb[i], 0)
#    
#    rgb_block[int(py),int(px),0] = np.nansum(hi4pi[477:626,int(py),int(px)]) # vlsr 10 to 100 km/s
#    rgb_block[int(py),int(px),1] = np.nansum(hi4pi[465:477,int(py),int(px)]) # vlsr -10 to 10 km/s
#    rgb_block[int(py),int(px),2] = np.nansum(hi4pi[315:465,int(py),int(px)]) # vlsr -100 to -10 km/s

# Set value of null cells to be average of the 2 cells next to it. Normally you don't need these next 10 lines, but the HI4PI
# dataset was missing several columns of data, I average the fluxes from cells around those missing data to fill in the blanks.
column_null = rgb_block.any(axis=0)[:,0]
row_null = rgb_block.any(axis=1)[:,0]
for i in range(rgb_block.shape[1]):
    if column_null[i] == False and i > 2 and i < rgb_block.shape[1]-2:
        # Use the commented line to average 4 cells instead of 2. It shouldn't change the final value too much
        # rgb_block[:,i,:] = 0.25 * (rgb_block[:,i+2,:] + rgb_block[:,i+1,:] + rgb_block[:,i-1,:] + rgb_block[:,i-2,:])
        rgb_block[:,i,:] = 0.5 * (rgb_block[:,i+1,:] + rgb_block[:,i-1,:])
for j in range(rgb_block.shape[0]):
    if row_null[j] == False and j > 2 and j < rgb_block.shape[0]-2:
        # Use the commented line to average 4 cells instead of 2. It shouldn't change the final value too much
        # rgb_block[j,:,:] = 0.25 * (rgb_block[j+2,:,:] + rgb_block[j+1,:,:] + rgb_block[j-1,:,:] + rgb_block[j-2,:,:])
        rgb_block[j,:,:] = 0.5 * (rgb_block[j+1,:,:] + rgb_block[j-1,:,:])

# Now we normalize the color channels so that they don't oversaturate, and convert each channel to an array in range [0,255]
rgb_grid = rgb_block
mean_sky = 150 #from: np.mean(rgb_block[:,:,:][np.nonzero(rgb_block[:,:,:])])
rgb_grid[:,:,0] = rgb_grid[:,:,0] / mean_sky * 40
rgb_grid[:,:,1] = rgb_grid[:,:,1] / mean_sky * 40
rgb_grid[:,:,2] = rgb_grid[:,:,2] / mean_sky * 40
rgb_grid[rgb_grid >= 256] = 255.999
rgb_grid = rgb_grid.astype(np.uint8)
# rgb_grid = np.flip(rgb_grid,0)


image = PIL.Image.fromarray(rgb_grid,'RGB')
image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)

#Remove black edge around the image (also specific to this dataset, since there is some padding from the datacube)
right = int(hi4pi_wcs.all_world2pix(patch_gl[-1],patch_gb[0],0)[0])
left = int(hi4pi_wcs.all_world2pix(patch_gl[0],patch_gb[0],0)[0])
top = int(hi4pi_wcs.all_world2pix(patch_gl[0],patch_gb[0],0)[1])
bottom = int(hi4pi_wcs.all_world2pix(patch_gl[0],patch_gb[-1],0)[1])
cropped_im = image.transpose(PIL.Image.FLIP_TOP_BOTTOM).crop((left,top,right,bottom)).transpose(PIL.Image.FLIP_TOP_BOTTOM)
plt.imshow(cropped_im)

# Uncomment to save image
# cropped_im.save(r'structure3_HI.jpg')

