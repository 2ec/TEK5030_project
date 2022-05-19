import cv2
import numpy as np
import image_plotter
import matplotlib.pyplot as plt
import networkx as nx
import map_segmentation
import osmnx as ox


def get_affine_transformation(img, pts1, pts2):
    rows,cols, *_ = img.shape
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(src=img, M=M, dsize=(cols, rows))

    return dst

def merge_images():
    pass

def rotate_crop_img(img, pts1, pts2, plot=False):    
    dst = get_affine_transformation(img, pts1, pts2)
    new_img = dst[550:1300, 900:1650].copy()
    if plot:
        plt.subplot(121)
        plt.imshow(img)
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(new_img)
        plt.title('Output')
        plt.show()

    return new_img


def image_to_graph(img:np.ndarray) -> nx.MultiGraph:
    x, y, _ = img.shape

    if x > y:
        img = img[:y, :y, 0]
    else:
        img = img[:x, :x, 0]

    G = nx.from_numpy_matrix(img, create_using=nx.MultiGraph)
    return G

def main():
    img = cv2.imread("dataset/map_segmented_roads/blindern_roads.png")
    plot = True

    # Rotate
    pts1 = np.float32([[55,55],[355,55],[555,350],[1550,1650]])
    pts2 = np.float32([[0,0],[300,0],[500,300],[1500,1600]])
    rotated = rotate_crop_img(img, pts1, pts2)

    if plot:
        plt.subplot(121)
        plt.imshow(img)
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(rotated)
        plt.title('Rotated')
        plt.show()

    # Unrotate 
    x, y, *_ = rotated.shape
    rotated_boarder = cv2.rectangle(rotated, (0,0), (x,y), color=(255, 0, 0), thickness=30)
    unrotate = np.zeros(img.shape, dtype=np.uint8)
    unrotate[550:1300, 900:1650] = rotated_boarder.copy()
    unrotate = get_affine_transformation(unrotate, pts2, pts1)

    if plot:
        plt.subplot(121)
        plt.imshow(img)
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(unrotate)
        plt.title('Unrotated')
        plt.show()

    overlay = image_plotter.get_overlay_img(img, unrotate)
    plt.imshow(overlay)
    plt.show()


def main1():
    img = cv2.imread("dataset/map_segmented_roads/blindern_roads.png")
    graph = image_to_graph(img)
    graph = nx.Graph(graph)

    center_point = (59.9433832, 10.727962) # Blindern
    dist = 800
    G = map_segmentation.get_road_img_from_center_point(center_point, dist=dist, edge_linewidth=2.0, show=False)
    G = nx.Graph(G)

    print(nx.faster_could_be_isomorphic(G, G.copy()))


def main3():
    ifi = (59.9435754, 10.7181494)
    dist = 200
    img_ifi = cv2.imread("dataset/satellite/blindern_ifi.png")
    
    G = map_segmentation.get_road_img_from_center_point(center_point=ifi, dist=dist, edge_linewidth=2.0, show=False, road_type="drive")
    buildings = map_segmentation.get_buildings_from_center_point(center_point=ifi, dist=dist)
    nodes, edges = ox.graph_to_gdfs(G)

    
    ax1 = plt.subplot(121)
    ax1.set_facecolor("black")
    buildings.plot(ax=ax1, facecolor="khaki", alpha=1.0,)
    edges.plot(ax=ax1, linewidth=2, edgecolor="white")
    plt.title('ifi')

    ax2 = plt.subplot(122)
    ax2.imshow(img_ifi)
    plt.title('Roads')
    plt.show()
    
    pts1 = np.float32([
        [10.721105, 59.9441596],
        [10.718055, 59.944496],
        [10.722278, 59.943155],
        [10.719949, 59.944371]])
    pts2 = np.float32([
        [995.0, 290.0],
        [504.7, 120.0],
        [1208.6, 632.1],
        [850.3, 200.0]])

    minx, miny, maxx, maxy = edges.total_bounds
    x_diff = int(maxx - minx)
    y_diff = int(maxy - miny)

    y, x, *_ = img_ifi.shape
    rotated_boarder = cv2.rectangle(img_ifi, (0,0), (x,y), color=(255, 0, 0), thickness=30)
    unrotate = np.zeros((x_diff, y_diff, 3), dtype=np.uint8)
    print(unrotate.shape)
    unrotate[11378-825:11378, 7807-1440:7807] = rotated_boarder.copy()
    unrotate = get_affine_transformation(unrotate, pts2, pts1)
    
    plt.imshow(unrotate)
    plt.show()

if __name__ == "__main__":
    # main()
    # main1()
    main3()