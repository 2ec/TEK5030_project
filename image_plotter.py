import numpy as np
import cv2

def load_image(filepath:str) -> np.ndarray: 
    img = cv2.imread(filepath)
    return img


def plot_images(imgs:list, titles:list) -> None:
    """
    Plots multiple images at once.
    
    Parameters: 
        imgs    (list): A list of images on np.ndarray format.
        titles  (list): A list of titles corresponding to the images in 'imgs'.
    """
    for i, img in enumerate(imgs):
        cv2.imshow(titles[i], img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def extend_image_channels(img:np.ndarray, num_chanels:int):
    x, y, *_ = img.shape
    new_img = np.zeros((x, y, num_chanels), dtype=img.dtype)
    for i in range(num_chanels):
        new_img[:, :, i] = img
    return new_img


def get_overlay_img(img1:np.ndarray, img2:np.ndarray, alpha:int=0.5):
    overlay = img1.copy()
    output = img2.copy()
    new_img = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return new_img