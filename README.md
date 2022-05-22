# TEK5030_project

## How to match images with maps
To install the required libraries to run our code, please use 
```
pip install -r requirements.txt
```

Our main file is merge_images.py. Please run this file to see the results. If you want to change images and maps being merged, please change filenames in the code. 

Please note that some files are deprecated versions of old code in this folder, and some files are required to make SuperGlue work. 

The report is also found in this zipped folder. Please refer to it for details on the project. 

## How to install OSMnx
This is used to download, extract and show roads from maps from OpenStreetMap.
Be in your chosen virtual Conda environment.
```
conda config --prepend channels conda-forge
conda install osmnx
```