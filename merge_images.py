import cv2
import numpy as np
import image_plotter
import matplotlib.pyplot as plt
import networkx as nx
import map_segmentation
import osmnx as ox
# from SuperGluePretrainedNetwork import get_keypoint_matches


def get_affine_transformation(img, pts1, pts2):
    rows,cols, *_ = img.shape
    
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(src=img, M=M, dsize=(cols, rows))

    return dst


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


def merge_images(img1, img2, match_points):
    img1_match = []
    img2_match = []

    for matches in match_points:
        point1, point2 = matches
        img1_match.append(point1)
        img2_match.append(point2)

    img1_match = np.array(img1_match, dtype=np.float32)
    img2_match = np.array(img2_match, dtype=np.float32)

    if img1.shape >= img2.shape:
        smallest_img = img2.copy()
        largest_img = img1.copy()
    else:
        smallest_img = img1.copy()
        largest_img = img2.copy()

    unrotate = np.zeros(largest_img.shape, dtype=np.uint8)
    
    x, y, *_ = smallest_img.shape
    unrotate[0:x, 0:y] = smallest_img.copy()
    unrotate_warped = get_affine_transformation(unrotate, img2_match, img1_match)

    overlay = image_plotter.get_overlay_img(largest_img, unrotate_warped)
    plt.imshow(overlay)
    plt.show()

def main4():
    kjeller_fly_medium = cv2.imread("dataset/map/kjeller_flyfoto_medium.png", )
    kjeller_kart_lite = cv2.imread("dataset/map/kjeller_kart_lite.png")

    kjeller_fly_medium = cv2.cvtColor(kjeller_fly_medium, cv2.COLOR_BGR2RGB)
    kjeller_kart_lite = cv2.cvtColor(kjeller_kart_lite, cv2.COLOR_BGR2RGB)

    if True:
        plt.subplot(121)
        plt.imshow(kjeller_fly_medium)
        plt.title('kjeller_fly_medium')
        plt.subplot(122)
        plt.imshow(kjeller_kart_lite)
        plt.title('kjeller_kart_lite')
        plt.show()

    # (kjeller_flyfoto_medium, kjeller_kart_lite)
    # ([1002., 518.], [258., 634.])
    # ([1064., 661.], [326., 777.]),
    # ([757., 684.], [13., 797.]),  
    # ([1081., 557.], [337., 672.]),
    # ([984., 101.], [240., 220.]),,
    match_points = [
                ([1370., 876.], [629., 987.]),
                ([760., 142.], [28., 256.]),
                ([753., 750.], [16., 870.]),
                ([1247., 260.], [504., 376.])
            ]
    fly_match = []
    map_match = []

    for matches in match_points:
        point1, point2 = matches
        fly_match.append(point1)
        map_match.append(point2)

    fly_match = np.array(fly_match, dtype=np.float32)
    map_match = np.array(map_match, dtype=np.float32)

    x, y, *_ = kjeller_kart_lite.shape
    #rotated_boarder = cv2.rectangle(kjeller_kart_lite, (0,0), (x,y), color=(255, 0, 0), thickness=30)
    unrotate = np.zeros(kjeller_fly_medium.shape, dtype=np.uint8)
    
    x, y, *_ = kjeller_kart_lite.shape
    unrotate[0:x, 0:y] = kjeller_kart_lite.copy()
    kjeller_kart_lite_warp = get_affine_transformation(unrotate, map_match, fly_match)

    print(kjeller_fly_medium.shape, kjeller_kart_lite_warp.shape)
    if True:
        plt.subplot(121)
        plt.imshow(kjeller_fly_medium)
        plt.title('Input')
        plt.subplot(122)
        plt.imshow(kjeller_kart_lite_warp)
        plt.title('Unrotated')
        plt.show()

    overlay = image_plotter.get_overlay_img(kjeller_fly_medium, kjeller_kart_lite_warp)
    plt.imshow(overlay)
    plt.show()


def main5():
    match_points = [
                ([1370., 876.], [629., 987.]), # R
                ([760., 142.], [28., 256.]), 
                ([753., 750.], [16., 870.]),
                ([1247., 260.], [504., 376.])
            ]
    
    kjeller_fly_medium = cv2.imread("dataset/map/kjeller_flyfoto_medium.png")
    kjeller_kart_lite = cv2.imread("dataset/map/kjeller_kart_lite.png")

    kjeller_fly_medium = cv2.cvtColor(kjeller_fly_medium, cv2.COLOR_BGR2RGB)
    kjeller_kart_lite = cv2.cvtColor(kjeller_kart_lite, cv2.COLOR_BGR2RGB)

    merge_images(kjeller_fly_medium, kjeller_kart_lite, match_points)


if __name__ == "__main__":
    # main()
    # main1()
    # main3()
    # main4()
    main5()
    