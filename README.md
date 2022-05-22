# TEK5030_project

## How to match images with maps
To install the required libraries to run our code, please use 
```
pip install -r requirements.txt
```

Our main file is merge_images.py. Please run this file to see the results. 

Please note that some files are deprecated versions of old code in this folder, and some files are required to make SuperGlue work. 

The report is also found in this zipped folder. Please refer to it for details on the project. 

## Pipeline of the code and how to run
Run 
``` python merge_images.py ```
to run the code with predefined images.
This calls further on "get_matchpoints()" from get_keypoint_matches.py,  which runs

&emsp;- match_image, from match_images_test.py, thats a modification of SuperGlues match_images

&emsp;- get_keypoints(), which extracts and sorts the best keypoints given in return from match_image's run_code()

merge_images(), then uses finds the homography, warps it and merges/overlays the images. 

* To test the code with other images, these should be specified in the main function in merge_images.py. Note the code has only been testet with images from Kartverkets Norgeskart, so please use the spesified aerial and "drawn" maps from this source for the best results. 

## File explaination
If you want to change images and maps being merged, please change filenames in the code.
All usable image should be put inside the "dataset" folder.

Segmented_unet.py was used to extract the roads from aerial photos. 
