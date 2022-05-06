import osmnx as ox
import networkx as nx
import cv2
import numpy as np
import matplotlib.pyplot as plt
import image_plotter

def segment_map_road(img:np.ndarray, max_value=255) -> np.ndarray:
    zeros = np.zeros(img.shape[:2], dtype=img.dtype)
    ones = np.full(img.shape[:2], max_value, dtype=img.dtype)
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    
    seg_img = np.where((b == g) & (b == r) & (b > 120) & (b < 190), ones, zeros)
    return seg_img


def morpho_clean_img(img:np.ndarray, kernel_size=3):
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size,kernel_size))
    kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
  
    blurred = cv2.medianBlur(img,3)
    opening = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel_cross)
    closing = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel_circle)

    diff = img - blurred
    return closing, opening, blurred, diff

def extract_roads_from_image(img, plot_imgs=False):
    seg_img = segment_map_road(img)
    seg_img_three_channels = image_plotter.extend_image_channels(seg_img, 3)
    overlay_img = image_plotter.get_overlay_img(seg_img_three_channels, img, alpha=0.5)
    close, open, blured, diff = morpho_clean_img(seg_img, kernel_size=3)
    
    if plot_imgs:
        images = [img, seg_img, close, open, blured, diff, overlay_img]
        titles = ["original", "seg", "close", "open", "blured", "diff", "overlay"]
        image_plotter.plot_images(images, titles)
    return open

def get_road_img_from_center_point(center_point:tuple, dist:int=800, edge_linewidth:float=2.5, road_type:str="drive", show:bool=True, save_filename:str=None) -> nx.MultiDiGraph:
    """
    https://towardsdatascience.com/making-artistic-maps-with-python-9d37f5ea8af0
    
    Gets an image 

    Parameters:
        center point (tuple)    : A touple given on the form (latitude, longitude).
        dist (int)              : How many meters in radius from the center point to get information from.
        edge_linewidth (float)  : How thick the lines drawn over roads should be.
        road_type (str)         : What type of road to get. Choose from "all_private", "all", "bike", "drive", "drive_service", "walk".
        show (bool)             : Whether to show the plotted image of roads.
        save_filename (str)     : What the filename of the saved file should be. If left by default (None) it will not save.

    Returns:
        G (networkx.MultiDiGraph) : A graph over the chosen road structure for the given area.
    """
    
    G = ox.graph_from_point(center_point, dist=dist, retain_all=True, simplify=True, network_type=road_type)
    save = False if save_filename is None else True

    """
    u = []
    v = []
    key = []
    data = []
    for uu, vv, kkey, ddata in G.edges(keys=True, data=True):
        u.append(uu)
        v.append(vv)
        key.append(kkey)
        data.append(ddata)   
    """
    if show or save:
        fig, ax = ox.plot_graph(G, node_size=0,
                                dpi=300, bgcolor = "#000000",
                                save=save, filepath=save_filename, show=show, edge_color="#FFFFFF",
                                edge_linewidth=edge_linewidth, edge_alpha=1)
    
    return G



def get_k_shortest_paths(G:nx.MultiDiGraph, origin_point:tuple, destination_point:tuple, center_point, dist, k=1, plot=True, save_filename=None) -> None:
    # https://www.geosnips.com/blogpost/osmnx-handling-street-networks
    save = False if save_filename is None else True

    origin_node = ox.nearest_nodes(G, X=origin_point[1], Y=origin_point[0])
    destination_node = ox.nearest_nodes(G, X=destination_point[1], Y=destination_point[0])
    bbox = ox.utils_geo.bbox_from_point(point=center_point, dist=dist)

    if k > 1:
        routes = ox.k_shortest_paths(G, origin_node, destination_node, k=k, weight="length")
        fig, ax = ox.plot_graph_routes(G, list(routes), bbox=bbox, route_colors="r", route_linewidth=2, edge_linewidth=2.0, node_size=0, save=save, filepath=save_filename)
    else:
        route = ox.shortest_path(G, origin_node,destination_node)
        fig, ax = ox.plot_graph_route(G, route, bbox=bbox, route_color="r", route_linewidth=6, edge_linewidth=2.0, node_size=0, bgcolor="k", save=save, filepath=save_filename)
    
    

if __name__ == "__main__":
    filepath = "dataset/map/blindern_kart.png"
    img = image_plotter.load_image(filepath)
    # morpho_open = extract_roads_from_image(img, plot_imgs=True)
 
    center_point = (59.9433832, 10.727962) # Blindern
    dist = 800
    G = get_road_img_from_center_point(center_point, dist=dist, edge_linewidth=2.0, save_filename="blindern_roads.png", show=False)
    
    ullevaal_stadion = (59.9488169, 10.7318353)
    blindern_studenterhjem = (59.9403866, 10.7205299)
    # get_k_shortest_paths(G, origin_point=ullevaal_stadion, destination_point=blindern_studenterhjem, k=1, center_point=center_point, dist=dist, save_filename="shortest_path_ullevaal_stadion_blindern_studenterhjem.png")