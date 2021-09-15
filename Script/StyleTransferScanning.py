# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 09:17:22 2021

@author: Trung Ha
"""

import os
import tensorflow as tf
import PIL

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
    
    if int(long_dim.numpy()) > 2000:
        max_dim = 2000
    elif int(long_dim.numpy()) < 500:
        max_dim = 500
    else:
        max_dim = int(long_dim.numpy())
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img,int(long_dim.numpy()),int(short_dim.numpy()) #int(min(new_shape).numpy())

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
reference_img,ref_dmax,ref_dmin = load_img('structure10.jpg')
milky_way_image,mw_dmax,mw_dmin = load_img('MW ha cart_0.jpg')

dtheta = 5 #shifting frame by this amount of degrees
dpixel = dtheta * 4

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


for i in range(int(mw_dmax/dpixel+1)):
    print('x: {}'.format(i*dpixel))
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
        
norm_dot_ratio_rescaled = norm_to_dot_product_matrix / np.max(norm_to_dot_product_matrix) 
        
max_pos = np.unravel_index(np.argmax(norm_to_dot_product_matrix),dot_product_matrix.shape)
print('Pixel position of max grid: {}'.format([[max_pos[0]*dpixel,max_pos[0]*dpixel+ref_dmax],[max_pos[1]*dpixel,max_pos[1]*dpixel+ref_dmin]]))
plt.imshow(mw_ghost[0])

pixel_dict = {}
max_instance = 1

for i in range(norm_dot_ratio_rescaled.shape[0]):
    for j in range(norm_dot_ratio_rescaled.shape[1]):
        if norm_dot_ratio_rescaled[i][j] > 0.9:
            x = [i*dpixel,i*dpixel+ref_dmax,i*dpixel+ref_dmax,i*dpixel,i*dpixel]
            y = [j*dpixel,j*dpixel,j*dpixel+ref_dmin,j*dpixel+ref_dmin,j*dpixel]
            #plt.plot(x,y,color='purple',linewidth=0.5,alpha=0.7)
            
            for k in range(i*dpixel, i*dpixel+ref_dmax):
                for l in range(j*dpixel, j*dpixel+ref_dmin):
                    coord_tuple = (k, l)
                    if coord_tuple in pixel_dict.keys():
                        pixel_dict[coord_tuple] = pixel_dict[coord_tuple] + 1
                        if pixel_dict[coord_tuple] > max_instance:
                            max_instance = pixel_dict[coord_tuple]
                    else:
                        pixel_dict[coord_tuple] = 1
           
x_list = []
y_list = []
for pixel in pixel_dict.keys():
    if pixel_dict[pixel] == max_instance:
        x_list.append(pixel[0])
        y_list.append(pixel[1])
        
plt.scatter(x_list, y_list, color='purple')
plt.xlim([0,mwg_dmax])
plt.axis('off')
plt.savefig('frame_grouping_s10.jpg',bbox_inches='tight',pad_inches = 0,dpi=600)