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

def get_homography_transformation(img, pts1, pts2):
    rows,cols, *_ = img.shape
    
    M, *_ = cv2.findHomography(pts1,pts2, method=cv2.RANSAC, ransacReprojThreshold=3)
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
    unrotate_warped = get_homography_transformation(unrotate, img2_match, img1_match) #get_affine_transformation(unrotate, img2_match, img1_match)

    overlay = image_plotter.get_overlay_img(largest_img, unrotate_warped)
    #plt.imshow(overlay)
    #plt.show()
    return overlay

def main4():
    kjeller_fly_medium = cv2.imread("dataset/map/kjeller_flyfoto_medium.png", )
    kjeller_kart_lite = cv2.imread("dataset/map/kjeller_kart_lite.png")

    kjeller_fly_medium = cv2.cvtColor(kjeller_fly_medium, cv2.COLOR_BGR2RGB)
    kjeller_kart_lite = cv2.cvtColor(kjeller_kart_lite, cv2.COLOR_BGR2RGB)

    match_points = gkm.get_matchpoints(kjeller_fly_medium, kjeller_kart_lite)
    print("----------------")
    print(match_points)

    if True:
        colours = ["*b", "*g", "*r", "*m"]
        plt.subplot(121)
        plt.imshow(kjeller_fly_medium)
        plt.title('kjeller_fly_medium')
        for i in range(len(match_points)):
            plt.plot(match_points[i][0][0], match_points[i][0][1], colours[i])
        plt.subplot(122)
        plt.imshow(kjeller_kart_lite)
        plt.title('kjeller_kart_lite')
        for i in range(len(match_points)):
            plt.plot(match_points[i][1][0], match_points[i][1][1], colours[i])
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

    overlay = image_plotter.get_overlay_img(kjeller_fly_medium, kjeller_kart_lite_warp, alpha=0.7)
    plt.imshow(overlay)
    plt.show()
    


def main5():
    # match_points = [
    #             ([1370., 876.], [629., 987.]),
    #             ([760., 142.], [28., 256.]), 
    #             ([753., 750.], [16., 870.]),
    #             ([1247., 260.], [504., 376.]),
    #             ([1002., 518.], [258., 634.]),
    #             ([1064., 661.], [326., 777.]),
    #             ([757., 684.], [13., 797.]),  
    #             ([1081., 557.], [337., 672.]),
    #             ([984., 101.], [240., 220.])
    #         ]

    match_points = [(np.array([757., 684.], dtype=np.float32), np.array([ 13., 797.], dtype=np.float32)), (np.array([1064.,  661.], dtype=np.float32), np.array([326., 777.], dtype=np.float32)), (np.array([1002.,  518.], dtype=np.float32), np.array([258., 634.], dtype=np.float32)), (np.array([1081.,  557.], dtype=np.float32), np.array([337., 672.], dtype=np.float32)), (np.array([950., 595.], dtype=np.float32), np.array([211., 707.], dtype=np.float32)), (np.array([1089.,  659.], dtype=np.float32), np.array([346., 774.], dtype=np.float32)), (np.array([1088.,  632.], dtype=np.float32), np.array([343., 748.], dtype=np.float32)), (np.array([917., 495.], dtype=np.float32), np.array([178., 613.], dtype=np.float32)), (np.array([878., 548.], dtype=np.float32), np.array([139., 665.], dtype=np.float32)), (np.array([902., 404.], dtype=np.float32), np.array([164., 520.], dtype=np.float32)), (np.array([796., 443.], dtype=np.float32), np.array([ 57., 561.], dtype=np.float32)), (np.array([883., 580.], dtype=np.float32), np.array([141., 696.], dtype=np.float32)), (np.array([904., 527.], dtype=np.float32), np.array([161., 640.], dtype=np.float32)), (np.array([1050.,  690.], dtype=np.float32), np.array([306., 805.], dtype=np.float32)), (np.array([984., 692.], dtype=np.float32), np.array([247., 803.], dtype=np.float32)), (np.array([823., 518.], dtype=np.float32), np.array([ 86., 633.], dtype=np.float32)), (np.array([1077.,  601.], dtype=np.float32), np.array([330., 716.], dtype=np.float32)), (np.array([832., 495.], dtype=np.float32), np.array([ 96., 612.], dtype=np.float32)), (np.array([841., 588.], dtype=np.float32), np.array([ 98., 704.], dtype=np.float32)), (np.array([791., 680.], dtype=np.float32), np.array([ 46., 795.], dtype=np.float32)), (np.array([1161.,  492.], dtype=np.float32), np.array([418., 606.], dtype=np.float32)), (np.array([851., 562.], dtype=np.float32), np.array([108., 676.], dtype=np.float32)), (np.array([841., 534.], dtype=np.float32), np.array([ 97., 648.], dtype=np.float32)), (np.array([790., 462.], dtype=np.float32), np.array([ 48., 579.], dtype=np.float32)), (np.array([1064.,  636.], dtype=np.float32), np.array([321., 754.], dtype=np.float32)), (np.array([885., 564.], dtype=np.float32), np.array([148., 679.], dtype=np.float32)), (np.array([898., 501.], dtype=np.float32), np.array([154., 616.], dtype=np.float32)), (np.array([1177.,  558.], dtype=np.float32), np.array([432., 673.], dtype=np.float32)), (np.array([851., 426.], dtype=np.float32), np.array([113., 543.], dtype=np.float32)), (np.array([861., 585.], dtype=np.float32), np.array([116., 703.], dtype=np.float32)), (np.array([1084.,  721.], dtype=np.float32), np.array([342., 839.], dtype=np.float32)), (np.array([1322.,  468.], dtype=np.float32), np.array([576., 582.], dtype=np.float32)), (np.array([1166.,  516.], dtype=np.float32), np.array([422., 631.], dtype=np.float32)), (np.array([876., 597.], dtype=np.float32), np.array([130., 713.], dtype=np.float32)), (np.array([869., 454.], dtype=np.float32), np.array([130., 568.], dtype=np.float32)), (np.array([1022.,  431.], dtype=np.float32), np.array([278., 547.], dtype=np.float32)), (np.array([815., 695.], dtype=np.float32), np.array([ 71., 807.], dtype=np.float32)), (np.array([957., 628.], dtype=np.float32), np.array([215., 744.], dtype=np.float32)), (np.array([1097.,  528.], dtype=np.float32), np.array([354., 642.], dtype=np.float32)), (np.array([806., 210.], dtype=np.float32), np.array([ 73., 338.], dtype=np.float32)), (np.array([1117.,  571.], dtype=np.float32), np.array([373., 684.], dtype=np.float32)), (np.array([855., 700.], dtype=np.float32), np.array([116., 815.], dtype=np.float32)), (np.array([871., 612.], dtype=np.float32), np.array([128., 728.], dtype=np.float32)), (np.array([1224.,  560.], dtype=np.float32), np.array([491., 680.], dtype=np.float32)), (np.array([967., 338.], dtype=np.float32), np.array([223., 454.], dtype=np.float32)), (np.array([985., 428.], dtype=np.float32), np.array([241., 544.], dtype=np.float32)), (np.array([1319.,  234.], dtype=np.float32), np.array([586., 352.], dtype=np.float32)), (np.array([1106.,  572.], dtype=np.float32), np.array([361., 687.], dtype=np.float32)), (np.array([1319.,  485.], dtype=np.float32), np.array([575., 601.], dtype=np.float32)), (np.array([826., 667.], dtype=np.float32), np.array([ 78., 783.], dtype=np.float32)), (np.array([857., 404.], dtype=np.float32), np.array([123., 520.], dtype=np.float32)), (np.array([761., 707.], dtype=np.float32), np.array([ 22., 825.], dtype=np.float32)), (np.array([1196.,  582.], dtype=np.float32), np.array([452., 695.], dtype=np.float32)), (np.array([1019.,  463.], dtype=np.float32), np.array([280., 576.], dtype=np.float32)), (np.array([761., 462.], dtype=np.float32), np.array([ 18., 575.], dtype=np.float32)), (np.array([962., 682.], dtype=np.float32), np.array([229., 796.], dtype=np.float32)), (np.array([1036.,  650.], dtype=np.float32), np.array([288., 761.], dtype=np.float32)), (np.array([774., 507.], dtype=np.float32), np.array([ 31., 620.], dtype=np.float32)), (np.array([1021.,  640.], dtype=np.float32), np.array([275., 753.], dtype=np.float32)), (np.array([1032.,  441.], dtype=np.float32), np.array([287., 559.], dtype=np.float32)), (np.array([872., 425.], dtype=np.float32), np.array([129., 541.], dtype=np.float32)), (np.array([859., 727.], dtype=np.float32), np.array([114., 838.], dtype=np.float32)), (np.array([1236.,  175.], dtype=np.float32), np.array([496., 300.], dtype=np.float32)), (np.array([750., 611.], dtype=np.float32), np.array([  9., 731.], dtype=np.float32)), (np.array([938., 614.], dtype=np.float32), np.array([202., 731.], dtype=np.float32)), (np.array([921., 212.], dtype=np.float32), np.array([178., 329.], dtype=np.float32)), (np.array([1120.,  594.], dtype=np.float32), np.array([377., 710.], dtype=np.float32)), (np.array([1050.,  617.], dtype=np.float32), np.array([313., 727.], dtype=np.float32)), (np.array([982., 748.], dtype=np.float32), np.array([242., 865.], dtype=np.float32)), (np.array([807., 488.], dtype=np.float32), np.array([ 79., 603.], dtype=np.float32)), (np.array([905., 457.], dtype=np.float32), np.array([170., 570.], dtype=np.float32)), (np.array([889., 321.], dtype=np.float32), np.array([147., 437.], dtype=np.float32)), (np.array([764., 735.], dtype=np.float32), np.array([ 27., 853.], dtype=np.float32)), (np.array([978., 659.], dtype=np.float32), np.array([239., 776.], dtype=np.float32)), (np.array([1292.,  475.], dtype=np.float32), np.array([547., 589.], dtype=np.float32)), (np.array([1010.,  360.], dtype=np.float32), np.array([277., 471.], dtype=np.float32)), (np.array([855., 455.], dtype=np.float32), np.array([112., 576.], dtype=np.float32)), (np.array([1154.,  152.], dtype=np.float32), np.array([403., 266.], dtype=np.float32)), (np.array([1314.,  346.], dtype=np.float32), np.array([570., 460.], dtype=np.float32)), (np.array([756., 546.], dtype=np.float32), np.array([ 10., 660.], dtype=np.float32)), (np.array([1114.,  150.], dtype=np.float32), np.array([386., 270.], dtype=np.float32)), (np.array([980., 805.], dtype=np.float32), np.array([243., 923.], dtype=np.float32)), (np.array([1318.,  416.], dtype=np.float32), np.array([597., 527.], dtype=np.float32)), (np.array([901., 432.], dtype=np.float32), np.array([165., 546.], dtype=np.float32)), (np.array([809., 448.], dtype=np.float32), np.array([ 79., 564.], dtype=np.float32)), (np.array([1164.,  423.], dtype=np.float32), np.array([422., 533.], dtype=np.float32)), (np.array([1327.,  209.], dtype=np.float32), np.array([597., 338.], dtype=np.float32)), (np.array([1060.,  539.], dtype=np.float32), np.array([321., 657.], dtype=np.float32)), (np.array([833., 192.], dtype=np.float32), np.array([ 88., 309.], dtype=np.float32)), (np.array([1155.,  562.], dtype=np.float32), np.array([413., 677.], dtype=np.float32)), (np.array([1287.,  171.], dtype=np.float32), np.array([541., 283.], dtype=np.float32)), (np.array([1315.,  248.], dtype=np.float32), np.array([586., 371.], dtype=np.float32)), (np.array([1171.,  347.], dtype=np.float32), np.array([404., 462.], dtype=np.float32)), (np.array([1230.,  190.], dtype=np.float32), np.array([497., 313.], dtype=np.float32)), (np.array([1277.,  491.], dtype=np.float32), np.array([531., 608.], dtype=np.float32)), (np.array([1062.,  566.], dtype=np.float32), np.array([325., 677.], dtype=np.float32)), (np.array([1258.,  654.], dtype=np.float32), np.array([516., 772.], dtype=np.float32)), (np.array([873., 322.], dtype=np.float32), np.array([132., 439.], dtype=np.float32)), (np.array([1211.,  656.], dtype=np.float32), np.array([477., 778.], dtype=np.float32)), (np.array([1064.,  280.], dtype=np.float32), np.array([317., 408.], dtype=np.float32)), (np.array([1158.,  276.], dtype=np.float32), np.array([416., 399.], dtype=np.float32)), (np.array([1339.,  461.], dtype=np.float32), np.array([593., 576.], dtype=np.float32)), (np.array([1342.,  286.], dtype=np.float32), np.array([605., 405.], dtype=np.float32)), (np.array([1017.,  395.], dtype=np.float32), np.array([273., 509.], dtype=np.float32)), (np.array([870., 498.], dtype=np.float32), np.array([136., 602.], dtype=np.float32)), (np.array([886., 808.], dtype=np.float32), np.array([141., 919.], dtype=np.float32)), (np.array([1053.,  401.], dtype=np.float32), np.array([308., 509.], dtype=np.float32)), (np.array([1280.,  406.], dtype=np.float32), np.array([534., 522.], dtype=np.float32)), (np.array([1333.,  255.], dtype=np.float32), np.array([612., 373.], dtype=np.float32)), (np.array([1223.,  356.], dtype=np.float32), np.array([489., 477.], dtype=np.float32)), (np.array([968., 525.], dtype=np.float32), np.array([223., 641.], dtype=np.float32)), (np.array([1329.,  557.], dtype=np.float32), np.array([583., 671.], dtype=np.float32)), (np.array([1044.,  217.], dtype=np.float32), np.array([279., 357.], dtype=np.float32)), (np.array([871., 366.], dtype=np.float32), np.array([136., 480.], dtype=np.float32)), (np.array([1359.,  288.], dtype=np.float32), np.array([623., 409.], dtype=np.float32)), (np.array([982., 626.], dtype=np.float32), np.array([240., 735.], dtype=np.float32)), (np.array([1223.,  283.], dtype=np.float32), np.array([477., 407.], dtype=np.float32)), (np.array([1097.,  585.], dtype=np.float32), np.array([353., 712.], dtype=np.float32)), (np.array([784., 778.], dtype=np.float32), np.array([ 38., 894.], dtype=np.float32)), (np.array([1251.,  321.], dtype=np.float32), np.array([532., 442.], dtype=np.float32)), (np.array([1238.,  432.], dtype=np.float32), np.array([512., 548.], dtype=np.float32)), (np.array([1170.,  623.], dtype=np.float32), np.array([443., 744.], dtype=np.float32)), (np.array([1157.,  350.], dtype=np.float32), np.array([404., 476.], dtype=np.float32)), (np.array([980., 257.], dtype=np.float32), np.array([237., 372.], dtype=np.float32)), (np.array([960., 794.], dtype=np.float32), np.array([226., 916.], dtype=np.float32)), (np.array([1264.,  155.], dtype=np.float32), np.array([520., 272.], dtype=np.float32)), (np.array([1287.,  250.], dtype=np.float32), np.array([543., 367.], dtype=np.float32)), (np.array([1136.,  483.], dtype=np.float32), np.array([389., 599.], dtype=np.float32)), (np.array([1021.,  588.], dtype=np.float32), np.array([290., 698.], dtype=np.float32)), (np.array([1045.,  454.], dtype=np.float32), np.array([314., 565.], dtype=np.float32)), (np.array([845., 199.], dtype=np.float32), np.array([ 98., 331.], dtype=np.float32)), (np.array([1198.,  500.], dtype=np.float32), np.array([451., 624.], dtype=np.float32)), (np.array([1264.,  183.], dtype=np.float32), np.array([520., 303.], dtype=np.float32)), (np.array([800., 501.], dtype=np.float32), np.array([ 54., 615.], dtype=np.float32)), (np.array([832., 232.], dtype=np.float32), np.array([ 92., 349.], dtype=np.float32)), (np.array([867., 476.], dtype=np.float32), np.array([130., 587.], dtype=np.float32)), (np.array([1082.,  774.], dtype=np.float32), np.array([344., 892.], dtype=np.float32)), (np.array([1009.,  266.], dtype=np.float32), np.array([271., 384.], dtype=np.float32)), (np.array([761., 295.], dtype=np.float32), np.array([ 30., 422.], dtype=np.float32)), (np.array([1111.,  266.], dtype=np.float32), np.array([366., 383.], dtype=np.float32)), (np.array([1040.,  360.], dtype=np.float32), np.array([297., 478.], dtype=np.float32)), (np.array([1342.,  311.], dtype=np.float32), np.array([599., 424.], dtype=np.float32)), (np.array([1232.,  394.], dtype=np.float32), np.array([496., 510.], dtype=np.float32)), (np.array([706.,  30.], dtype=np.float32), np.array([  9., 206.], dtype=np.float32)), (np.array([870., 779.], dtype=np.float32), np.array([137., 888.], dtype=np.float32)), (np.array([948., 649.], dtype=np.float32), np.array([212., 762.], dtype=np.float32)), (np.array([902., 767.], dtype=np.float32), np.array([153., 888.], dtype=np.float32)), (np.array([937., 284.], dtype=np.float32), np.array([194., 400.], dtype=np.float32)), (np.array([1102.,  334.], dtype=np.float32), np.array([358., 450.], dtype=np.float32)), (np.array([1006.,  563.], dtype=np.float32), np.array([262., 682.], dtype=np.float32)), (np.array([879., 225.], dtype=np.float32), np.array([141., 341.], dtype=np.float32)), (np.array([943., 193.], dtype=np.float32), np.array([222., 317.], dtype=np.float32)), (np.array([997., 251.], dtype=np.float32), np.array([260., 359.], dtype=np.float32)), (np.array([1364.,  406.], dtype=np.float32), np.array([619., 522.], dtype=np.float32)), (np.array([819., 890.], dtype=np.float32), np.array([122., 979.], dtype=np.float32)), (np.array([1051.,  506.], dtype=np.float32), np.array([315., 625.], dtype=np.float32)), (np.array([1029.,  804.], dtype=np.float32), np.array([306., 922.], dtype=np.float32)), (np.array([939., 218.], dtype=np.float32), np.array([215., 341.], dtype=np.float32)), (np.array([821., 336.], dtype=np.float32), np.array([ 83., 460.], dtype=np.float32)), (np.array([1001.,  300.], dtype=np.float32), np.array([258., 411.], dtype=np.float32)), (np.array([1086.,  138.], dtype=np.float32), np.array([345., 256.], dtype=np.float32)), (np.array([1124.,  760.], dtype=np.float32), np.array([390., 890.], dtype=np.float32)), (np.array([1205.,  320.], dtype=np.float32), np.array([458., 439.], dtype=np.float32)), (np.array([940., 887.], dtype=np.float32), np.array([208., 999.], dtype=np.float32)), (np.array([918., 479.], dtype=np.float32), np.array([186., 594.], dtype=np.float32)), (np.array([1233.,  138.], dtype=np.float32), np.array([438., 249.], dtype=np.float32)), (np.array([839., 215.], dtype=np.float32), np.array([118., 355.], dtype=np.float32)), (np.array([911., 360.], dtype=np.float32), np.array([185., 478.], dtype=np.float32)), (np.array([879., 286.], dtype=np.float32), np.array([144., 396.], dtype=np.float32)), (np.array([973., 305.], dtype=np.float32), np.array([229., 415.], dtype=np.float32)), (np.array([1184.,  859.], dtype=np.float32), np.array([482., 999.], dtype=np.float32)), (np.array([892., 784.], dtype=np.float32), np.array([183., 897.], dtype=np.float32)), (np.array([772., 433.], dtype=np.float32), np.array([ 27., 549.], dtype=np.float32)), (np.array([771., 795.], dtype=np.float32), np.array([ 27., 909.], dtype=np.float32)), (np.array([1225.,  374.], dtype=np.float32), np.array([501., 495.], dtype=np.float32)), (np.array([763., 181.], dtype=np.float32), np.array([ 20., 301.], dtype=np.float32)), (np.array([1230.,  251.], dtype=np.float32), np.array([503., 375.], dtype=np.float32)), (np.array([761., 308.], dtype=np.float32), np.array([ 28., 434.], dtype=np.float32)), (np.array([984.,  99.], dtype=np.float32), np.array([203., 275.], dtype=np.float32)), (np.array([1287.,  372.], dtype=np.float32), np.array([546., 475.], dtype=np.float32)), (np.array([773., 627.], dtype=np.float32), np.array([ 28., 749.], dtype=np.float32)), (np.array([1173.,  682.], dtype=np.float32), np.array([422., 834.], dtype=np.float32)), (np.array([1074.,  577.], dtype=np.float32), np.array([340., 690.], dtype=np.float32)), (np.array([740., 215.], dtype=np.float32), np.array([ 19., 334.], dtype=np.float32)), (np.array([1250.,  824.], dtype=np.float32), np.array([541., 980.], dtype=np.float32)), (np.array([1334.,  539.], dtype=np.float32), np.array([589., 654.], dtype=np.float32)), (np.array([1187.,  702.], dtype=np.float32), np.array([438., 849.], dtype=np.float32)), (np.array([924., 684.], dtype=np.float32), np.array([180., 799.], dtype=np.float32)), (np.array([1073.,  392.], dtype=np.float32), np.array([335., 533.], dtype=np.float32)), (np.array([1235.,  591.], dtype=np.float32), np.array([508., 715.], dtype=np.float32)), (np.array([1246.,  416.], dtype=np.float32), np.array([498., 530.], dtype=np.float32)), (np.array([1236.,  362.], dtype=np.float32), np.array([506., 480.], dtype=np.float32)), (np.array([833., 430.], dtype=np.float32), np.array([110., 559.], dtype=np.float32)), (np.array([819., 747.], dtype=np.float32), np.array([ 63., 866.], dtype=np.float32)), (np.array([809., 291.], dtype=np.float32), np.array([ 80., 393.], dtype=np.float32)), (np.array([1122.,  432.], dtype=np.float32), np.array([386., 536.], dtype=np.float32)), (np.array([1114.,  782.], dtype=np.float32), np.array([382., 903.], dtype=np.float32)), (np.array([1028.,  754.], dtype=np.float32), np.array([279., 868.], dtype=np.float32)), (np.array([949., 418.], dtype=np.float32), np.array([203., 531.], dtype=np.float32)), (np.array([769.,  20.], dtype=np.float32), np.array([ 52., 214.], dtype=np.float32)), (np.array([818., 462.], dtype=np.float32), np.array([ 74., 579.], dtype=np.float32)), (np.array([763., 285.], dtype=np.float32), np.array([ 33., 395.], dtype=np.float32)), (np.array([1033.,  394.], dtype=np.float32), np.array([291., 510.], dtype=np.float32)), (np.array([1359.,  486.], dtype=np.float32), np.array([627., 594.], dtype=np.float32)), (np.array([1049.,  752.], dtype=np.float32), np.array([325., 877.], dtype=np.float32)), (np.array([1312.,  792.], dtype=np.float32), np.array([570., 910.], dtype=np.float32)), (np.array([1047.,  114.], dtype=np.float32), np.array([302., 231.], dtype=np.float32)), (np.array([740., 233.], dtype=np.float32), np.array([ 10., 354.], dtype=np.float32)), (np.array([1207.,  470.], dtype=np.float32), np.array([480., 590.], dtype=np.float32)), (np.array([1315.,  503.], dtype=np.float32), np.array([603., 629.], dtype=np.float32)), (np.array([951., 565.], dtype=np.float32), np.array([228., 684.], dtype=np.float32)), (np.array([999., 733.], dtype=np.float32), np.array([284., 851.], dtype=np.float32)), (np.array([1216.,  606.], dtype=np.float32), np.array([471., 716.], dtype=np.float32)), (np.array([853., 788.], dtype=np.float32), np.array([120., 910.], dtype=np.float32)), (np.array([1365.,  321.], dtype=np.float32), np.array([615., 442.], dtype=np.float32)), (np.array([1276.,  640.], dtype=np.float32), np.array([532., 756.], dtype=np.float32)), (np.array([1083.,  271.], dtype=np.float32), np.array([339., 389.], dtype=np.float32)), (np.array([858., 535.], dtype=np.float32), np.array([116., 656.], dtype=np.float32)), (np.array([819., 379.], dtype=np.float32), np.array([ 91., 487.], dtype=np.float32)), (np.array([1247.,  381.], dtype=np.float32), np.array([518., 503.], dtype=np.float32)), (np.array([907., 672.], dtype=np.float32), np.array([161., 790.], dtype=np.float32)), (np.array([1034.,  143.], dtype=np.float32), np.array([245., 325.], dtype=np.float32)), (np.array([1182.,  668.], dtype=np.float32), np.array([432., 817.], dtype=np.float32)), (np.array([1249.,  609.], dtype=np.float32), np.array([535., 738.], dtype=np.float32)), (np.array([990., 785.], dtype=np.float32), np.array([268., 911.], dtype=np.float32)), (np.array([824., 270.], dtype=np.float32), np.array([ 91., 421.], dtype=np.float32)), (np.array([822., 559.], dtype=np.float32), np.array([ 77., 673.], dtype=np.float32)), (np.array([1205.,  727.], dtype=np.float32), np.array([494., 853.], dtype=np.float32)), (np.array([1337.,  577.], dtype=np.float32), np.array([593., 692.], dtype=np.float32)), (np.array([971., 775.], dtype=np.float32), np.array([219., 885.], dtype=np.float32)), (np.array([737., 270.], dtype=np.float32), np.array([ 16., 387.], dtype=np.float32)), (np.array([935., 456.], dtype=np.float32), np.array([207., 551.], dtype=np.float32)), (np.array([1322.,  686.], dtype=np.float32), np.array([533., 782.], dtype=np.float32)), (np.array([885., 412.], dtype=np.float32), np.array([145., 523.], dtype=np.float32)), (np.array([943., 530.], dtype=np.float32), np.array([198., 663.], dtype=np.float32)), (np.array([931., 327.], dtype=np.float32), np.array([178., 426.], dtype=np.float32)), (np.array([1188.,  399.], dtype=np.float32), np.array([443., 513.], dtype=np.float32)), (np.array([970., 138.], dtype=np.float32), np.array([194., 308.], dtype=np.float32)), (np.array([923., 716.], dtype=np.float32), np.array([186., 822.], dtype=np.float32)), (np.array([1284.,  266.], dtype=np.float32), np.array([538., 384.], dtype=np.float32)), (np.array([1178.,  288.], dtype=np.float32), np.array([448., 407.], dtype=np.float32)), (np.array([743., 146.], dtype=np.float32), np.array([ 31., 258.], dtype=np.float32)), (np.array([1185.,  378.], dtype=np.float32), np.array([423., 482.], dtype=np.float32)), (np.array([1068.,  487.], dtype=np.float32), np.array([318., 585.], dtype=np.float32)), (np.array([836., 452.], dtype=np.float32), np.array([ 92., 583.], dtype=np.float32)), (np.array([837., 782.], dtype=np.float32), np.array([109., 888.], dtype=np.float32)), (np.array([874., 826.], dtype=np.float32), np.array([154., 932.], dtype=np.float32)), (np.array([1260.,  424.], dtype=np.float32), np.array([532., 548.], dtype=np.float32)), (np.array([975., 459.], dtype=np.float32), np.array([248., 572.], dtype=np.float32)), (np.array([780., 109.], dtype=np.float32), np.array([ 51., 237.], dtype=np.float32)), (np.array([915., 380.], dtype=np.float32), np.array([187., 493.], dtype=np.float32)), (np.array([1048.,  336.], dtype=np.float32), np.array([314., 451.], dtype=np.float32)), (np.array([1297.,  315.], dtype=np.float32), np.array([489., 419.], dtype=np.float32)), (np.array([1251.,  204.], dtype=np.float32), np.array([524., 333.], dtype=np.float32)), (np.array([773., 492.], dtype=np.float32), np.array([ 17., 609.], dtype=np.float32)), (np.array([1011.,  500.], dtype=np.float32), np.array([264., 609.], dtype=np.float32)), (np.array([978., 575.], dtype=np.float32), np.array([236., 709.], dtype=np.float32)), (np.array([1359.,  572.], dtype=np.float32), np.array([611., 678.], dtype=np.float32)), (np.array([1300.,  675.], dtype=np.float32), np.array([568., 789.], dtype=np.float32)), (np.array([1289.,  739.], dtype=np.float32), np.array([495., 910.], dtype=np.float32)), (np.array([1029.,  230.], dtype=np.float32), np.array([284., 345.], dtype=np.float32)), (np.array([906.,  95.], dtype=np.float32), np.array([123., 242.], dtype=np.float32)), (np.array([1335.,  341.], dtype=np.float32), np.array([588., 450.], dtype=np.float32)), (np.array([1076.,  440.], dtype=np.float32), np.array([360., 579.], dtype=np.float32)), (np.array([1194.,  244.], dtype=np.float32), np.array([400., 362.], dtype=np.float32)), (np.array([794., 321.], dtype=np.float32), np.array([ 76., 432.], dtype=np.float32)), (np.array([1068.,  322.], dtype=np.float32), np.array([311., 435.], dtype=np.float32)), (np.array([776., 880.], dtype=np.float32), np.array([ 42., 994.], dtype=np.float32)), (np.array([978.,  76.], dtype=np.float32), np.array([238., 194.], dtype=np.float32)), (np.array([1121.,  508.], dtype=np.float32), np.array([381., 613.], dtype=np.float32)), (np.array([796., 277.], dtype=np.float32), np.array([ 64., 376.], dtype=np.float32)), (np.array([782., 568.], dtype=np.float32), np.array([ 35., 683.], dtype=np.float32)), (np.array([1191.,  688.], dtype=np.float32), np.array([409., 875.], dtype=np.float32)), (np.array([1335.,  365.], dtype=np.float32), np.array([611., 481.], dtype=np.float32)), (np.array([939., 496.], dtype=np.float32), np.array([212., 594.], dtype=np.float32)), (np.array([779., 302.], dtype=np.float32), np.array([ 69., 413.], dtype=np.float32)), (np.array([909., 333.], dtype=np.float32), np.array([164., 450.], dtype=np.float32)), (np.array([851., 505.], dtype=np.float32), np.array([112., 633.], dtype=np.float32)), (np.array([1221.,  512.], dtype=np.float32), np.array([485., 638.], dtype=np.float32)), (np.array([1140.,  348.], dtype=np.float32), np.array([376., 461.], dtype=np.float32)), (np.array([1274.,  760.], dtype=np.float32), np.array([521., 888.], dtype=np.float32)), (np.array([783., 362.], dtype=np.float32), np.array([ 21., 474.], dtype=np.float32)), (np.array([722., 226.], dtype=np.float32), np.array([ 10., 343.], dtype=np.float32)), (np.array([920., 287.], dtype=np.float32), np.array([176., 399.], dtype=np.float32)), (np.array([1277.,  290.], dtype=np.float32), np.array([464., 420.], dtype=np.float32)), (np.array([1193.,  648.], dtype=np.float32), np.array([459., 767.], dtype=np.float32)), (np.array([1068.,  224.], dtype=np.float32), np.array([335., 375.], dtype=np.float32)), (np.array([1200.,  264.], dtype=np.float32), np.array([464., 380.], dtype=np.float32)), (np.array([800.,  28.], dtype=np.float32), np.array([ 57., 144.], dtype=np.float32)), (np.array([1306.,   30.], dtype=np.float32), np.array([414.,  61.], dtype=np.float32))]
    
    kjeller_fly_medium = cv2.imread("dataset/map/kjeller_flyfoto_medium.png")
    kjeller_kart_lite = cv2.imread("dataset/map/kjeller_kart_lite.png")

    kjeller_fly_medium = cv2.cvtColor(kjeller_fly_medium, cv2.COLOR_BGR2RGB)
    kjeller_kart_lite = cv2.cvtColor(kjeller_kart_lite, cv2.COLOR_BGR2RGB)

    overlay = merge_images(kjeller_fly_medium, kjeller_kart_lite, match_points)
    plt.imsave("overlay_homography_all_points.png", overlay, dpi=400.)
    

if __name__ == "__main__":
    # main()
    # main1()
    # main3()
    # main4()
    main5()
    