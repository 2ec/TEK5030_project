# TEK5030_project

## How to match images with maps
To install the required libraries to run our code, please use 
```
pip install -r requirements.txt
```

Our main file is merge_images.py. Please run this file to see the results. 

Please note that some files are deprecated versions of old code in this folder, and some files are required to make SuperGlue work. 

The report is also found in this zipped folder. Please refer to it for details on the project. 

## Pipeline of the code
Run 
``` python merge_images.py ```
this calls further on "get_matchpoints()" from get_keypoint_matches.py,  which runs

&emsp;- match_image, from match_images_test.py, thats a modification of SuperGlues match_images

&emsp;- get_keypoints(), which extracts and sorts the best keypoints given in return from match_image's run_code()

merge_images(), then uses finds the homography, warps it and merges/overlays the images. 

## File explaination
If you want to change images and maps being merged, please change filenames in the code.
All usable image should be put inside the "dataset" folder.
## How to install OSMnx
This is used to download, extract and show roads from maps from OpenStreetMap.
Be in your chosen virtual Conda environment.
```
conda config --prepend channels conda-forge
conda install osmnx
```
