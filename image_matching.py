import numpy as np
import cv2
import matplotlib.pyplot as plt

background = cv2.imread("dataset/map/kart_oversikt_blindern.png")
background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
face = cv2.imread("dataset/map/blindern_vestgrensa.png")
face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
height, width, channels = face.shape
# print(background.shape)
# print(face.shape)

methods = ['cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR']

for x in methods:
    bg_copy = background.copy()
    method = eval(x)
    result = cv2.matchTemplate(bg_copy, face, method)

    #Henter max og min verdi og lokasjon
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]: #fungerer annerleder enn de andre
        top_left = min_loc 
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + width, top_left[1] + height)

    cv2.rectangle(bg_copy, top_left, bottom_right, 255, 10)

    plt.subplot(121)
    plt.imshow(result)
    plt.title("Template matching resultat")

    plt.subplot(122)
    plt.imshow(bg_copy)
    plt.title('resultatbilde')
    plt.suptitle(x)
    plt.show()
# plt.imshow(background)
# plt.show()
