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
import cv2

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

def get_skeleton(img):
    #Source: http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html

    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
 
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
    
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True

    return skel


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
    input_arr = input_arr[:,centre_l:centre_r].astype(np.uint8)
    input_arr = resize(input_arr, (IMAGE_HEIGHT, IMAGE_WIDTH), anti_aliasing=True, preserve_range=True).astype(np.uint)
    #f, ax = plt.subplots(1,3)
    #ax[0].imshow(input_arr)
    #ax[0].set_title("Original")
    input_arr = np.array([input_arr]) 

    #model = load_model("C:/Users/Sande/OneDrive/Courses/tek5030/TEK5030_project/models/road_mapper_final.h5", 
    #                   custom_objects={'soft_dice_loss': soft_dice_loss(), 'iou_coef': iou_coef()})

    model = load_model(directory + "/models/road_mapper_final.h5",compile=False)
    
    predictions = model.predict(input_arr, verbose=1)
    #ax[1].imshow(predictions[0])
    #ax[1].set_title("Prediction")
    print(predictions[0].shape)

    thresh_val = 0.01
    prediction_threshold = (predictions > thresh_val).astype(np.uint8)
    #ax[2].imshow(prediction_threshold[0])
    #ax[2].set_title("t = "+ str(thresh_val))

    f, ax = plt.subplots(1,2)

    prediction_threshold = prediction_threshold * 255
    ax[0].imshow(prediction_threshold[0])

    skeleton = get_skeleton(prediction_threshold[0])
    ax[1].imshow(skeleton)

    #plt.imshow(skeleton)

    imsave("skeleton.png", skeleton)
    plt.show()
    
   
