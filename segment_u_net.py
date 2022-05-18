#Imports
import tensorflow as tf 
from tensorflow.keras.models import Model, load_model
import numpy as np
import pandas as pd
import pickle
import h5py
from matplotlib import pyplot as plt 
from skimage.io import imread, imshow, imread_collection, concatenate_images, imsave
from skimage.transform import resize
import glob
import os

def segment_airphoto(image_name):

    #Source: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    IMAGE_HEIGHT = IMAGE_WIDTH = 256
    NUM_CHANNELS = 3

    directory = os.getcwd()
    print(directory)
    image_path = directory + "/dataset/satellite/" + image_name
    print(image_path)
    
    image = tf.keras.utils.load_img(path=image_path)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    N, M = input_arr.shape[0], input_arr.shape[1]
    centre_l, centre_r = (M-N)//2, M-((M-N)//2)
    input_arr = input_arr[:,centre_l:centre_r].astype(np.uint)
    input_arr = resize(input_arr, (IMAGE_HEIGHT, IMAGE_WIDTH), anti_aliasing=True, preserve_range=True).astype(np.uint)
    input_arr = np.array([input_arr]) 

    model = load_model(directory + "/models/road_mapper_final.h5",compile=False)
    
    predictions = model.predict(input_arr, verbose=1)

    thresh_val = 0.1
    prediction_threshold = (predictions > thresh_val).astype(np.uint8)
    prediction_threshold = prediction_threshold * 255

    return prediction_threshold

if __name__ == "__main__":

    #Source: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2

    IMAGE_HEIGHT = IMAGE_WIDTH = 256
    NUM_CHANNELS = 3

    directory = os.getcwd()
    print(directory)
    image_path = directory + "/dataset/satellite/sving.png"
    print(image_path)
    
    image = tf.keras.utils.load_img(path=image_path)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    N, M = input_arr.shape[0], input_arr.shape[1]
    centre_l, centre_r = (M-N)//2, M-((M-N)//2)
    input_arr = input_arr[:,centre_l:centre_r].astype(np.uint)
    input_arr = resize(input_arr, (IMAGE_HEIGHT, IMAGE_WIDTH), anti_aliasing=True, preserve_range=True).astype(np.uint)
    f, ax = plt.subplots(1,3)
    ax[0].imshow(input_arr)
    input_arr = np.array([input_arr]) 

    #model = load_model("C:/Users/Sande/OneDrive/Courses/tek5030/TEK5030_project/models/road_mapper_final.h5", 
    #                   custom_objects={'soft_dice_loss': soft_dice_loss(), 'iou_coef': iou_coef()})

    model = load_model(directory + "/models/road_mapper_final.h5",compile=False)
    
    predictions = model.predict(input_arr, verbose=1)
    ax[1].imshow(predictions[0])
    print(predictions[0].shape)

    thresh_val = 0.1
    prediction_threshold = (predictions > thresh_val).astype(np.uint8)
    ax[2].imshow(prediction_threshold[0])
    #prediction_threshold = tf.squeeze(prediction_threshold)
    print(prediction_threshold[0].shape)
    prediction_threshold = prediction_threshold * 255

    imsave("satelite_thresholded.png", prediction_threshold[0])
    plt.show()
    
   