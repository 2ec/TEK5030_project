import operator
import os
import cv2
import numpy as np
from scipy import ndimage
from match_images_test import match_image

def get_keypoints(npz, num_keypoints=4):
    """
    Input: 
        npzpath: path to npz matching file
        num_keypoints: number of keypoints
    Output:
        keypoints: list with tuples of coordinates to mapping points within the image
    
    """

    # npz = np.load(npzpath)
    # print(npz.files)
    #['keypoints0', 'keypoints1', 'matches', 'match_confidence']

    kp1 = npz['keypoints1']
    kp0 = npz['keypoints0']
    matches = npz['matches']
    match_confidence = npz['match_confidence']

    total = 0
    true_matches = matches > 0
    for i in true_matches:
        if i:
            total += 1
    if num_keypoints == "all":
        num_keypoints = total
    if total < num_keypoints:
        num_keypoints = total

    zipped = zip(kp0, matches, match_confidence )
    res = sorted(zipped, key = operator.itemgetter(2))[::-1] #sorts by matchconfidence

    keypoints = []
    for i in range(num_keypoints):
        best_kp0, index_kp1, confidence = res[i]
        corresponding_point = kp1[index_kp1]
        keypoints.append((best_kp0, corresponding_point))

    return keypoints


def crop_image(imagepath, bottom_percent, top_percent):
    out_of_scale_factor = 0.8
    image = cv2.imread(imagepath)
    height, width, channels = image.shape
    new_bottom_height = int(height * bottom_percent)
    new_top_height = int(height * top_percent * out_of_scale_factor)
    new_bottom_width = int(width * bottom_percent)
    new_top_width = int(width * top_percent)
    print(new_bottom_height, new_bottom_width, new_top_height, new_top_width)

    new_image = image[new_bottom_height:new_top_height, new_bottom_width:new_top_width]
    return new_image

def rotate_image(image, rot_deg):

    return ndimage.rotate(image, rot_deg)

def resize_image(img, scale_percent=80):
    # scale_percent = 60 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def get_matchpoints(img1, img2, num_keypoints=4):
    superglue = match_image(img1, img2)
    output = superglue.run_code()
    keypoints = get_keypoints(output, num_keypoints=num_keypoints)
    return keypoints


if __name__ == "__main__":
    os_path = os.getcwd()
    dir_path = os.path.join(os_path,"assets/test_images/")



    # image = cv2.imread(imagepath1)
    # percent = 0.7
    # image = crop_image(imagepath, 1- percent, percent)
    rot_deg = 10
    # # image = cv2.rotate(image, cv2.cv2.ROTATE_45_CLOCKWISE)
    # kernel = (2,2)
    # blur_image = cv2.blur(image, kernel)
    # name = "scale"
    name = "rotate"
    # # scale_percent = 60
    # # image = resize_image(image, scale_percent=scale_percent)
    # # name = f"resize_{scale_percent}"¨
    # image = rotate_image(image, rot_deg)
    # cv2.imwrite(f"{dir_path}{image_name1[:-4]}_{name}_{rot_deg}.png", image)
    path = 'result_images/blindern1_blindern_scale_07_matches.npz'

    # points = get_keypoints(path)
    # print(points)
    image_name1 = "kjeller_flyfoto_medium.png"
    image_name2 = "kjeller_kart_lite.png"
    imagepath1 = os.path.join(dir_path, image_name1)
    imagepath2 = os.path.join(dir_path, image_name1)
    main_imagepath = os.path.join(dir_path, image_name1)
    side_imagepath = os.path.join(dir_path, image_name2)
    main_image = cv2.imread(main_imagepath)
    side_image = cv2.imread(side_imagepath)

    keypoints = get_matchpoints(main_image, side_image, "all")
    print(keypoints)
