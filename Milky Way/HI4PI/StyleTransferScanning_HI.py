# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:17:22 2021

@author: Trung Ha
"""

import os
os.chdir("F:\\OneDrive - UNT System\\RESEARCH_DOCS\\AI_Project\\StyleTransfer\\Script")
import tensorflow as tf
import PIL
from tqdm import tqdm

import imageio
import matplotlib.pyplot as plt
import numpy as np

def load_img(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    short_dim = min(shape)
    
    if int(long_dim.numpy()) > 4320:
        max_dim = 4320
    elif int(long_dim.numpy()) < 500:
        max_dim = 500
    else:
        max_dim = int(long_dim.numpy())
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img,int(long_dim.numpy()),int(short_dim.numpy()),float(scale.numpy()) #int(min(new_shape).numpy())

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

#%%
# Gram matrix on a small window scan that is shifted by x degrees each time
reference_img,ref_dmax,ref_dmin,scale_ref = load_img('structure2_HI.jpg')
milky_way_image,mw_dmax,mw_dmin,scale_mw = load_img('fullskyfin_HI.jpg')
mw_dmax, mw_dmin = mw_dmax*scale_mw, mw_dmin*scale_mw

dtheta = 5 #shifting frame by this amount of degrees
deg_pix = 12 # how many pixel per degree? 4 for WHAM, 12 for HI4PI
dpixel = dtheta * deg_pix 

style_extractor = vgg_layers(style_layers)
ref_style_output = [gram_matrix(style_output) for style_output in style_extractor(reference_img*255)]
ref_gram = ref_style_output[4].numpy().flatten()

# Create a ghost zone of reference image width to account for the boundary
mw_ghost = np.concatenate((milky_way_image,milky_way_image[:,:,:ref_dmax,:]),axis=2)
mwg_dmax = max(mw_ghost.shape)
mwg_dmin = mw_dmin
# plt.imshow(mw_ghost[0])

dot_product_matrix = np.zeros((int(mw_dmax/dpixel+1),int(mw_dmin/dpixel)))
norm_product_matrix = np.zeros((int(mw_dmax/dpixel+1),int(mw_dmin/dpixel)))
norm_to_dot_product_matrix = np.zeros((int(mw_dmax/dpixel+1),int(mw_dmin/dpixel)))


for i in tqdm(range(int(mw_dmax/dpixel+1))):
    # print('x: {}'.format(i*dpixel))
    for j in range(int(mw_dmin/dpixel)):
        mw_sector = mw_ghost[:,j*dpixel:j*dpixel+ref_dmin,i*dpixel:i*dpixel+ref_dmax,:]
        # print (mw_sector.shape)
        style_outputs = style_extractor(mw_sector)
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]
        milky_gram = style_outputs[4].numpy().flatten()
        # print('Dot product = {}'.format(np.dot(milky_gram, ref_gram)))
        dot_product_matrix[i][j] = np.dot(milky_gram, ref_gram)
        norm_product_matrix[i][j] = np.linalg.norm(milky_gram)*np.linalg.norm(ref_gram)
        norm_to_dot_product_matrix[i][j] = dot_product_matrix[i][j]/norm_product_matrix[i][j]
        
norm_dot_ratio_rescaled = norm_to_dot_product_matrix / np.nanmax(norm_to_dot_product_matrix) 

#%%
       
max_pos = np.unravel_index(np.argmax(norm_to_dot_product_matrix),dot_product_matrix.shape)
print('Pixel position of max grid: {}'.format([[max_pos[0]*dpixel,max_pos[0]*dpixel+ref_dmax],[max_pos[1]*dpixel,max_pos[1]*dpixel+ref_dmin]]))

# Pick top_N number of frames to display
top_N = 70
val_sim = np.zeros((top_N,))
idx_sim = np.argpartition(norm_dot_ratio_rescaled, norm_dot_ratio_rescaled.size - top_N, axis=None)[-top_N:]
pos_sim = np.column_stack(np.unravel_index(idx_sim, norm_dot_ratio_rescaled.shape))

for i in range(len(pos_sim)):
    val_sim[i] = norm_dot_ratio_rescaled[pos_sim[i][0],pos_sim[i][1]]
threshold = np.min(val_sim)
    
repeat_arg = np.zeros((mw_ghost.shape[2],mw_ghost.shape[1]))

for i in range(norm_dot_ratio_rescaled.shape[0]-int(ref_dmax/dpixel)+1):
    for j in range(norm_dot_ratio_rescaled.shape[1]):
        if norm_dot_ratio_rescaled[i][j] >= 0.9:#threshold:
            x = [i*dpixel,i*dpixel+ref_dmax,i*dpixel+ref_dmax,i*dpixel,i*dpixel]
            y = [j*dpixel,j*dpixel,j*dpixel+ref_dmin,j*dpixel+ref_dmin,j*dpixel]
            plt.plot(x,y,color='white',linewidth=0.8,alpha=0.7)
            repeat_arg[i*dpixel:i*dpixel+ref_dmax,j*dpixel:j*dpixel+ref_dmin] += 1
      
plt.imshow(mw_ghost[0])
plt.xlim([0,mwg_dmax])
plt.axis('off')
# plt.savefig('sim_s7_top70.jpg',bbox_inches='tight',pad_inches = 0,dpi=600)

# Overlay contours
import plotly.io as pio
pio.renderers.default='browser'
import plotly.graph_objects as go
import plotly.express as px
image = plt.imread('fullskyfin_HI.jpg')
fig = px.imshow(image, color_continuous_scale='gray')
fig.add_trace(go.Contour(
        z=np.transpose(repeat_arg),opacity=1,
        colorscale='Oranges',
        contours=dict(
            start=0,
            end=np.max(repeat_arg),
            size=2),
        contours_coloring='lines',
        line_width=2
    ))
fig.update_layout(yaxis_title='Latitude',
                  xaxis_title='Longitude',
                  xaxis = dict(ticktext= ['180','240','300','0','60','120','180'], tickvals= [0,240,480,720,960,1200,1440]),
                  yaxis = dict(ticktext= ['-90','-60','-30','0','30','60','90'], tickvals= [0,120,240,360,480,600,720]),
                  autosize=False,
                  width=mw_ghost.shape[2],
                  height=mw_ghost.shape[1])

fig.update_xaxes(showgrid=False, range=[0,max(image.shape)])
fig.update_yaxes(showgrid=False, scaleanchor='x', range=(mw_ghost.shape[1], 0))
fig.show()
 
# Plot frequency in 3d
# fig = go.Figure(data=[go.Surface(z=repeat_arg)])
# fig.update_layout(scene_aspectratio=dict(x=1,y=mw_ghost.shape[2]/mw_ghost.shape[1],z=1))
# fig.update_traces(contours_z=dict(show=True, usecolormap=True,
#                                   highlightcolor="limegreen", project_z=True))
# fig.update_layout(scene = dict(
#                     xaxis_title='Latitude',
#                     yaxis_title='Longitude',
#                     zaxis_title='Frequency of Similarity',
#                     yaxis = dict(
#                         ticktext= ['180','240','300','0','60','120','180'],
#                         tickvals= [0,240,480,720,960,1200,1440]),
#                     xaxis = dict(
#                         ticktext= ['-90','-60','-30','0','30','60','90'],
#                         tickvals= [0,120,240,360,480,600,720]),
#                     zaxis = dict(
#                         nticks=8, ticks='outside',
#                         tick0=0, tickwidth=4),),
#                     width=2000,
#                     margin=dict(r=20, b=10, l=10, t=10))
# fig.show()
        

