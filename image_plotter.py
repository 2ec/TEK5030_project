import numpy as np
import matplotlib.pyplot as plt
import osmnx as ox
from networkx import MultiDiGraph
import cv2
from geopandas import GeoDataFrame

def load_image(filepath:str) -> np.ndarray: 
    img = cv2.imread(filepath)
    return img

def plot_roads_buildings_shortest_path(G:MultiDiGraph, buildings:GeoDataFrame=None, route:list=None) -> None:
    nodes, edges = ox.graph_to_gdfs(G)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_facecolor("black")

    if buildings is not None:
        print("Buildings")
        buildings.plot(ax=ax, facecolor="khaki", alpha=1.0,)
    
    edges.plot(ax=ax, linewidth=2, edgecolor="white")
    
    """
    texts = []
    for _, edge in ox.graph_to_gdfs(G, nodes=False).fillna('').iterrows():
        c = edge['geometry'].centroid
        text = edge['name']

        if text not in texts:
            texts.append(text)
            ax.annotate(text, (c.x, c.y), c='w')
    """
    if route is not None:
        print("Route")
        ox.plot_graph_route(G, route, route_color="r", ax=ax, route_alpha=0.7)
    plt.tight_layout()
    plt.title("Roads")
    plt.savefig("dataset/map_segmented_roads/roads_buildings.png", dpi=700, bbox_inches='tight', pad_inches=0)
    plt.show()

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


def get_overlay_img(img1:np.ndarray, img2:np.ndarray, alpha:int=0.5) -> np.ndarray:
    overlay = img1.copy()
    output = img2.copy()
    new_img = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return new_img