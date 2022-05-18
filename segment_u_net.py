#Imports
import tensorflow as tf 
from tensorflow.keras.models import Model, load_model
import numpy as np
import pandas as pd
import pickle
import h5py
from matplotlib import pyplot as plt 
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
import glob



if __name__ == "__main__":

    #Source: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    from tensorflow.keras import backend as K
    def iou_coef(y_true, y_pred, smooth=1):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    
        return iou

    def dice_coef(y_true, y_pred, smooth = 1):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def soft_dice_loss(y_true, y_pred):
        return 1-dice_coef(y_true, y_pred)

    IMAGE_HEIGHT = IMAGE_WIDTH = 256
    NUM_CHANNELS = 3

    image_path = 'C:/Users/Sande/OneDrive/Courses/tek5030/TEK5030_project/dataset/satellite/gravel.png'

    
    image = tf.keras.utils.load_img(path=image_path)
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    N, M = input_arr.shape[0], input_arr.shape[1]
    centre_l, centre_r = (M-N)//2, M-((M-N)//2)
    input_arr = input_arr[:,centre_l:centre_r].astype(np.uint)
    input_arr = resize(input_arr, (IMAGE_HEIGHT, IMAGE_WIDTH), anti_aliasing=True, preserve_range=True).astype(np.uint)
    f, ax = plt.subplots(1,5)
    ax[0].imshow(input_arr)
    input_arr = np.array([input_arr]) 

    #model = load_model("C:/Users/Sande/OneDrive/Courses/tek5030/TEK5030_project/models/road_mapper_final.h5", 
    #                   custom_objects={'soft_dice_loss': soft_dice_loss(), 'iou_coef': iou_coef()})

    model = load_model("C:/Users/Sande/OneDrive/Courses/tek5030/TEK5030_project/models/road_mapper_final.h5",compile=False)
    
    predictions = model.predict(input_arr, verbose=1)
    ax[1].imshow(predictions[0])

    thresh_val = 0.1
    prediction_threshold = (predictions > thresh_val).astype(np.uint8)
    ax[2].imshow(prediction_threshold[0])

    thresh_val = 0.000005
    prediction_threshold2 = (predictions > thresh_val).astype(np.uint8)
    ax[3].imshow(prediction_threshold2[0])

    thresh_val = 0.000002
    prediction_threshold3 = (predictions > thresh_val).astype(np.uint8)
    ax[4].imshow(prediction_threshold3[0])

    plt.show()