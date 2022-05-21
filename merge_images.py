import cv2
import numpy as np
import image_plotter
import matplotlib.pyplot as plt
import networkx as nx
import map_segmentation
import osmnx as ox
import get_keypoint_matches as gkm


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

def main6():
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
    kjeller_fly_medium = cv2.imread("dataset/map/kjeller_kart_ny.png", )
    kjeller_kart_lite = cv2.imread("dataset/map/kjeller_flyfoto_ny.png")

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

    match_points = [(np.array([516., 579.], dtype=np.float32), np.array([ 96., 264.], dtype=np.float32)), (np.array([594., 555.], dtype=np.float32), np.array([173., 240.], dtype=np.float32)), (np.array([565., 561.], dtype=np.float32), np.array([142., 247.], dtype=np.float32)), (np.array([514., 567.], dtype=np.float32), np.array([ 93., 252.], dtype=np.float32)), (np.array([640., 583.], dtype=np.float32), np.array([212., 267.], dtype=np.float32)), (np.array([631., 599.], dtype=np.float32), np.array([207., 282.], dtype=np.float32)), (np.array([466., 641.], dtype=np.float32), np.array([ 44., 325.], dtype=np.float32)), (np.array([823., 587.], dtype=np.float32), np.array([397., 268.], dtype=np.float32)), (np.array([463., 576.], dtype=np.float32), np.array([ 38., 260.], dtype=np.float32)), (np.array([479., 651.], dtype=np.float32), np.array([ 58., 337.], dtype=np.float32)), (np.array([605., 476.], dtype=np.float32), np.array([184., 162.], dtype=np.float32)), (np.array([435., 581.], dtype=np.float32), np.array([ 14., 266.], dtype=np.float32)), (np.array([551., 604.], dtype=np.float32), np.array([125., 287.], dtype=np.float32)), (np.array([666., 580.], dtype=np.float32), np.array([236., 261.], dtype=np.float32)), (np.array([477., 638.], dtype=np.float32), np.array([ 57., 323.], dtype=np.float32)), (np.array([622., 504.], dtype=np.float32), np.array([196., 187.], dtype=np.float32)), (np.array([644., 597.], dtype=np.float32), np.array([223., 280.], dtype=np.float32)), (np.array([501., 685.], dtype=np.float32), np.array([ 80., 370.], dtype=np.float32)), (np.array([571., 568.], dtype=np.float32), np.array([151., 252.], dtype=np.float32)), (np.array([787., 482.], dtype=np.float32), np.array([368., 167.], dtype=np.float32)), (np.array([580., 559.], dtype=np.float32), np.array([159., 245.], dtype=np.float32)), (np.array([831., 540.], dtype=np.float32), np.array([407., 221.], dtype=np.float32)), (np.array([797., 586.], dtype=np.float32), np.array([375., 272.], dtype=np.float32)), (np.array([655., 470.], dtype=np.float32), np.array([232., 155.], dtype=np.float32)), (np.array([613., 591.], dtype=np.float32), np.array([180., 277.], dtype=np.float32)), (np.array([824., 657.], dtype=np.float32), np.array([392., 341.], dtype=np.float32)), (np.array([805., 572.], dtype=np.float32), np.array([385., 260.], dtype=np.float32)), (np.array([579., 502.], dtype=np.float32), np.array([156., 193.], dtype=np.float32)), (np.array([592., 564.], dtype=np.float32), np.array([172., 249.], dtype=np.float32)), (np.array([539., 635.], dtype=np.float32), np.array([113., 320.], dtype=np.float32)), (np.array([535., 648.], dtype=np.float32), np.array([109., 331.], dtype=np.float32)), (np.array([531., 575.], dtype=np.float32), np.array([112., 257.], dtype=np.float32)), (np.array([632., 562.], dtype=np.float32), np.array([204., 246.], dtype=np.float32)), (np.array([528., 635.], dtype=np.float32), np.array([100., 327.], dtype=np.float32)), (np.array([818., 606.], dtype=np.float32), np.array([397., 290.], dtype=np.float32)), (np.array([441., 456.], dtype=np.float32), np.array([ 17., 141.], dtype=np.float32)), (np.array([748., 681.], dtype=np.float32), np.array([330., 353.], dtype=np.float32)), (np.array([729., 561.], dtype=np.float32), np.array([300., 247.], dtype=np.float32)), (np.array([554., 529.], dtype=np.float32), np.array([136., 215.], dtype=np.float32)), (np.array([757., 648.], dtype=np.float32), np.array([325., 329.], dtype=np.float32)), (np.array([621., 583.], dtype=np.float32), np.array([199., 267.], dtype=np.float32)), (np.array([705., 608.], dtype=np.float32), np.array([285., 295.], dtype=np.float32)), (np.array([822., 643.], dtype=np.float32), np.array([393., 330.], dtype=np.float32)), (np.array([508., 550.], dtype=np.float32), np.array([ 85., 234.], dtype=np.float32)), (np.array([558., 580.], dtype=np.float32), np.array([135., 262.], dtype=np.float32)), (np.array([849., 719.], dtype=np.float32), np.array([407., 386.], dtype=np.float32)), (np.array([738., 567.], dtype=np.float32), np.array([309., 249.], dtype=np.float32)), (np.array([556., 511.], dtype=np.float32), np.array([136., 197.], dtype=np.float32)), (np.array([442., 518.], dtype=np.float32), np.array([ 21., 204.], dtype=np.float32)), (np.array([634., 550.], dtype=np.float32), np.array([203., 231.], dtype=np.float32)), (np.array([751., 668.], dtype=np.float32), np.array([334., 342.], dtype=np.float32)), (np.array([656., 594.], dtype=np.float32), np.array([234., 282.], dtype=np.float32)), (np.array([471., 621.], dtype=np.float32), np.array([ 42., 304.], dtype=np.float32)), (np.array([657., 704.], dtype=np.float32), np.array([221., 370.], dtype=np.float32)), (np.array([834., 588.], dtype=np.float32), np.array([405., 271.], dtype=np.float32)), (np.array([621., 608.], dtype=np.float32), np.array([192., 293.], dtype=np.float32)), (np.array([866., 684.], dtype=np.float32), np.array([436., 365.], dtype=np.float32)), (np.array([462., 546.], dtype=np.float32), np.array([ 36., 234.], dtype=np.float32)), (np.array([744., 732.], dtype=np.float32), np.array([310., 388.], dtype=np.float32)), (np.array([710., 549.], dtype=np.float32), np.array([285., 227.], dtype=np.float32)), (np.array([483., 585.], dtype=np.float32), np.array([ 61., 271.], dtype=np.float32)), (np.array([834., 552.], dtype=np.float32), np.array([411., 236.], dtype=np.float32)), (np.array([432., 513.], dtype=np.float32), np.array([ 12., 201.], dtype=np.float32)), (np.array([723., 495.], dtype=np.float32), np.array([305., 180.], dtype=np.float32)), (np.array([734., 629.], dtype=np.float32), np.array([311., 314.], dtype=np.float32)), (np.array([693., 606.], dtype=np.float32), np.array([273., 292.], dtype=np.float32)), (np.array([573., 416.], dtype=np.float32), np.array([153., 104.], dtype=np.float32)), (np.array([625., 495.], dtype=np.float32), np.array([193., 177.], dtype=np.float32)), (np.array([437., 605.], dtype=np.float32), np.array([ 15., 290.], dtype=np.float32)), (np.array([651., 461.], dtype=np.float32), np.array([226., 144.], dtype=np.float32)), (np.array([614., 493.], dtype=np.float32), np.array([181., 176.], dtype=np.float32)), (np.array([828., 608.], dtype=np.float32), np.array([409., 297.], dtype=np.float32)), (np.array([600., 506.], dtype=np.float32), np.array([180., 189.], dtype=np.float32)), (np.array([674., 602.], dtype=np.float32), np.array([250., 290.], dtype=np.float32)), (np.array([791., 466.], dtype=np.float32), np.array([372., 151.], dtype=np.float32)), (np.array([428., 726.], dtype=np.float32), np.array([  8., 388.], dtype=np.float32)), (np.array([548., 697.], dtype=np.float32), np.array([115., 376.], dtype=np.float32)), (np.array([617., 436.], dtype=np.float32), np.array([196., 121.], dtype=np.float32)), (np.array([608., 466.], dtype=np.float32), np.array([184., 150.], dtype=np.float32)), (np.array([441., 651.], dtype=np.float32), np.array([ 21., 338.], dtype=np.float32)), (np.array([729., 579.], dtype=np.float32), np.array([310., 265.], dtype=np.float32)), (np.array([772., 608.], dtype=np.float32), np.array([351., 293.], dtype=np.float32)), (np.array([433., 655.], dtype=np.float32), np.array([ 10., 336.], dtype=np.float32)), (np.array([587., 587.], dtype=np.float32), np.array([164., 271.], dtype=np.float32)), (np.array([804., 440.], dtype=np.float32), np.array([378., 130.], dtype=np.float32)), (np.array([634., 454.], dtype=np.float32), np.array([215., 139.], dtype=np.float32)), (np.array([715., 493.], dtype=np.float32), np.array([294., 178.], dtype=np.float32)), (np.array([550., 473.], dtype=np.float32), np.array([124., 148.], dtype=np.float32)), (np.array([686., 701.], dtype=np.float32), np.array([254., 368.], dtype=np.float32)), (np.array([700., 588.], dtype=np.float32), np.array([279., 275.], dtype=np.float32)), (np.array([666., 570.], dtype=np.float32), np.array([236., 253.], dtype=np.float32)), (np.array([784., 605.], dtype=np.float32), np.array([363., 289.], dtype=np.float32)), (np.array([427., 571.], dtype=np.float32), np.array([ 10., 251.], dtype=np.float32)), (np.array([688., 595.], dtype=np.float32), np.array([268., 281.], dtype=np.float32)), (np.array([426., 536.], dtype=np.float32), np.array([  9., 211.], dtype=np.float32)), (np.array([795., 563.], dtype=np.float32), np.array([371., 252.], dtype=np.float32)), (np.array([503., 604.], dtype=np.float32), np.array([ 71., 293.], dtype=np.float32)), (np.array([567., 581.], dtype=np.float32), np.array([149., 268.], dtype=np.float32)), (np.array([864., 662.], dtype=np.float32), np.array([429., 344.], dtype=np.float32)), (np.array([505., 461.], dtype=np.float32), np.array([ 92., 145.], dtype=np.float32)), (np.array([571., 485.], dtype=np.float32), np.array([147., 169.], dtype=np.float32)), (np.array([688., 535.], dtype=np.float32), np.array([271., 217.], dtype=np.float32)), (np.array([694., 571.], dtype=np.float32), np.array([267., 253.], dtype=np.float32)), (np.array([415., 707.], dtype=np.float32), np.array([ 12., 373.], dtype=np.float32)), (np.array([646., 456.], dtype=np.float32), np.array([221., 136.], dtype=np.float32)), (np.array([450., 713.], dtype=np.float32), np.array([ 38., 382.], dtype=np.float32)), (np.array([838., 600.], dtype=np.float32), np.array([420., 293.], dtype=np.float32)), (np.array([840., 658.], dtype=np.float32), np.array([410., 343.], dtype=np.float32)), (np.array([611., 603.], dtype=np.float32), np.array([177., 286.], dtype=np.float32)), (np.array([507., 540.], dtype=np.float32), np.array([ 74., 224.], dtype=np.float32)), (np.array([733., 496.], dtype=np.float32), np.array([312., 180.], dtype=np.float32)), (np.array([501., 709.], dtype=np.float32), np.array([ 75., 386.], dtype=np.float32)), (np.array([473., 592.], dtype=np.float32), np.array([ 48., 280.], dtype=np.float32)), (np.array([489., 455.], dtype=np.float32), np.array([ 68., 149.], dtype=np.float32)), (np.array([612., 451.], dtype=np.float32), np.array([179., 135.], dtype=np.float32)), (np.array([785., 452.], dtype=np.float32), np.array([362., 138.], dtype=np.float32)), (np.array([566., 400.], dtype=np.float32), np.array([145.,  85.], dtype=np.float32)), (np.array([573., 462.], dtype=np.float32), np.array([154., 140.], dtype=np.float32)), (np.array([524., 611.], dtype=np.float32), np.array([101., 303.], dtype=np.float32)), (np.array([762., 703.], dtype=np.float32), np.array([326., 369.], dtype=np.float32)), (np.array([770., 579.], dtype=np.float32), np.array([342., 254.], dtype=np.float32)), (np.array([508., 502.], dtype=np.float32), np.array([ 91., 182.], dtype=np.float32)), (np.array([756., 736.], dtype=np.float32), np.array([321., 389.], dtype=np.float32)), (np.array([718., 462.], dtype=np.float32), np.array([290., 150.], dtype=np.float32)), (np.array([656., 506.], dtype=np.float32), np.array([236., 190.], dtype=np.float32)), (np.array([644., 566.], dtype=np.float32), np.array([220., 250.], dtype=np.float32)), (np.array([520., 695.], dtype=np.float32), np.array([ 94., 389.], dtype=np.float32)), (np.array([678., 667.], dtype=np.float32), np.array([249., 348.], dtype=np.float32)), (np.array([609., 547.], dtype=np.float32), np.array([183., 229.], dtype=np.float32)), (np.array([826., 561.], dtype=np.float32), np.array([404., 247.], dtype=np.float32)), (np.array([610., 419.], dtype=np.float32), np.array([186., 103.], dtype=np.float32)), (np.array([572., 642.], dtype=np.float32), np.array([151., 327.], dtype=np.float32)), (np.array([459., 587.], dtype=np.float32), np.array([ 43., 276.], dtype=np.float32)), (np.array([445., 698.], dtype=np.float32), np.array([ 26., 384.], dtype=np.float32)), (np.array([833., 456.], dtype=np.float32), np.array([411., 141.], dtype=np.float32)), (np.array([821., 482.], dtype=np.float32), np.array([402., 166.], dtype=np.float32)), (np.array([604., 433.], dtype=np.float32), np.array([176., 110.], dtype=np.float32)), (np.array([553., 690.], dtype=np.float32), np.array([128., 374.], dtype=np.float32)), (np.array([673., 487.], dtype=np.float32), np.array([246., 171.], dtype=np.float32)), (np.array([495., 619.], dtype=np.float32), np.array([ 61., 298.], dtype=np.float32)), (np.array([574., 517.], dtype=np.float32), np.array([144., 200.], dtype=np.float32)), (np.array([544., 516.], dtype=np.float32), np.array([130., 204.], dtype=np.float32)), (np.array([859., 699.], dtype=np.float32), np.array([439., 382.], dtype=np.float32)), (np.array([477., 671.], dtype=np.float32), np.array([ 56., 357.], dtype=np.float32)), (np.array([739., 710.], dtype=np.float32), np.array([284., 376.], dtype=np.float32)), (np.array([786., 619.], dtype=np.float32), np.array([384., 310.], dtype=np.float32)), (np.array([577., 447.], dtype=np.float32), np.array([156., 132.], dtype=np.float32)), (np.array([544., 653.], dtype=np.float32), np.array([106., 359.], dtype=np.float32)), (np.array([453., 536.], dtype=np.float32), np.array([ 28., 227.], dtype=np.float32)), (np.array([708., 513.], dtype=np.float32), np.array([291., 199.], dtype=np.float32)), (np.array([697., 598.], dtype=np.float32), np.array([281., 284.], dtype=np.float32)), (np.array([699., 564.], dtype=np.float32), np.array([286., 247.], dtype=np.float32)), (np.array([568., 624.], dtype=np.float32), np.array([153., 317.], dtype=np.float32)), (np.array([477., 683.], dtype=np.float32), np.array([ 78., 380.], dtype=np.float32)), (np.array([825., 523.], dtype=np.float32), np.array([397., 203.], dtype=np.float32)), (np.array([735., 471.], dtype=np.float32), np.array([346., 162.], dtype=np.float32)), (np.array([548., 583.], dtype=np.float32), np.array([123., 263.], dtype=np.float32)), (np.array([732., 481.], dtype=np.float32), np.array([305., 166.], dtype=np.float32)), (np.array([410., 539.], dtype=np.float32), np.array([ 30., 211.], dtype=np.float32)), (np.array([706., 659.], dtype=np.float32), np.array([298., 343.], dtype=np.float32)), (np.array([527., 521.], dtype=np.float32), np.array([106., 206.], dtype=np.float32)), (np.array([444., 499.], dtype=np.float32), np.array([ 20., 186.], dtype=np.float32)), (np.array([774., 569.], dtype=np.float32), np.array([344., 246.], dtype=np.float32)), (np.array([809., 636.], dtype=np.float32), np.array([379., 326.], dtype=np.float32)), (np.array([536., 513.], dtype=np.float32), np.array([125., 194.], dtype=np.float32)), (np.array([742., 517.], dtype=np.float32), np.array([332., 198.], dtype=np.float32)), (np.array([522., 600.], dtype=np.float32), np.array([ 89., 288.], dtype=np.float32)), (np.array([660., 452.], dtype=np.float32), np.array([239., 135.], dtype=np.float32)), (np.array([646., 638.], dtype=np.float32), np.array([262., 336.], dtype=np.float32)), (np.array([742., 500.], dtype=np.float32), np.array([323., 185.], dtype=np.float32)), (np.array([674., 683.], dtype=np.float32), np.array([243., 364.], dtype=np.float32)), (np.array([827., 424.], dtype=np.float32), np.array([414., 109.], dtype=np.float32)), (np.array([814., 473.], dtype=np.float32), np.array([388., 161.], dtype=np.float32)), (np.array([640., 466.], dtype=np.float32), np.array([207., 151.], dtype=np.float32)), (np.array([721., 551.], dtype=np.float32), np.array([302., 236.], dtype=np.float32)), (np.array([417., 491.], dtype=np.float32), np.array([  8., 180.], dtype=np.float32)), (np.array([555., 562.], dtype=np.float32), np.array([122., 247.], dtype=np.float32)), (np.array([615., 525.], dtype=np.float32), np.array([195., 210.], dtype=np.float32)), (np.array([722., 604.], dtype=np.float32), np.array([292., 287.], dtype=np.float32)), (np.array([722., 450.], dtype=np.float32), np.array([334., 141.], dtype=np.float32)), (np.array([551., 643.], dtype=np.float32), np.array([118., 321.], dtype=np.float32)), (np.array([643., 654.], dtype=np.float32), np.array([200., 335.], dtype=np.float32)), (np.array([413., 697.], dtype=np.float32), np.array([  8., 365.], dtype=np.float32)), (np.array([781., 584.], dtype=np.float32), np.array([350., 271.], dtype=np.float32)), (np.array([581., 677.], dtype=np.float32), np.array([138., 377.], dtype=np.float32)), (np.array([731., 461.], dtype=np.float32), np.array([342., 145.], dtype=np.float32)), (np.array([421., 406.], dtype=np.float32), np.array([ 39., 101.], dtype=np.float32)), (np.array([704., 691.], dtype=np.float32), np.array([294., 361.], dtype=np.float32)), (np.array([526., 510.], dtype=np.float32), np.array([105., 195.], dtype=np.float32)), (np.array([462., 714.], dtype=np.float32), np.array([ 54., 391.], dtype=np.float32)), (np.array([853., 584.], dtype=np.float32), np.array([416., 269.], dtype=np.float32)), (np.array([718., 485.], dtype=np.float32), np.array([297., 169.], dtype=np.float32)), (np.array([534., 528.], dtype=np.float32), np.array([112., 213.], dtype=np.float32)), (np.array([715., 592.], dtype=np.float32), np.array([294., 278.], dtype=np.float32)), (np.array([817., 461.], dtype=np.float32), np.array([391., 145.], dtype=np.float32)), (np.array([812., 556.], dtype=np.float32), np.array([386., 241.], dtype=np.float32)), (np.array([764., 659.], dtype=np.float32), np.array([347., 343.], dtype=np.float32)), (np.array([428., 677.], dtype=np.float32), np.array([ 12., 357.], dtype=np.float32)), (np.array([452., 588.], dtype=np.float32), np.array([ 25., 268.], dtype=np.float32)), (np.array([452., 676.], dtype=np.float32), np.array([ 44., 337.], dtype=np.float32)), (np.array([886., 379.], dtype=np.float32), np.array([415.,  95.], dtype=np.float32)), (np.array([836., 563.], dtype=np.float32), np.array([414., 256.], dtype=np.float32)), (np.array([488., 654.], dtype=np.float32), np.array([ 37., 351.], dtype=np.float32)), (np.array([545., 482.], dtype=np.float32), np.array([128., 168.], dtype=np.float32)), (np.array([768., 494.], dtype=np.float32), np.array([353., 171.], dtype=np.float32)), (np.array([715., 711.], dtype=np.float32), np.array([294., 391.], dtype=np.float32)), (np.array([777., 451.], dtype=np.float32), np.array([385., 147.], dtype=np.float32)), (np.array([487., 607.], dtype=np.float32), np.array([ 66., 292.], dtype=np.float32)), (np.array([438., 365.], dtype=np.float32), np.array([22., 39.], dtype=np.float32)), (np.array([830., 435.], dtype=np.float32), np.array([388., 124.], dtype=np.float32)), (np.array([720., 657.], dtype=np.float32), np.array([307., 340.], dtype=np.float32)), (np.array([657., 617.], dtype=np.float32), np.array([237., 291.], dtype=np.float32)), (np.array([435., 413.], dtype=np.float32), np.array([13., 77.], dtype=np.float32)), (np.array([648., 479.], dtype=np.float32), np.array([224., 161.], dtype=np.float32)), (np.array([850., 413.], dtype=np.float32), np.array([424.,  96.], dtype=np.float32)), (np.array([523., 672.], dtype=np.float32), np.array([ 89., 362.], dtype=np.float32)), (np.array([849., 554.], dtype=np.float32), np.array([423., 258.], dtype=np.float32)), (np.array([424., 618.], dtype=np.float32), np.array([ 12., 302.], dtype=np.float32)), (np.array([791., 628.], dtype=np.float32), np.array([347., 305.], dtype=np.float32)), (np.array([461., 387.], dtype=np.float32), np.array([66., 91.], dtype=np.float32)), (np.array([695., 494.], dtype=np.float32), np.array([268., 168.], dtype=np.float32)), (np.array([663., 745.], dtype=np.float32), np.array([218., 390.], dtype=np.float32)), (np.array([490., 679.], dtype=np.float32), np.array([ 68., 369.], dtype=np.float32)), (np.array([708., 446.], dtype=np.float32), np.array([287., 133.], dtype=np.float32)), (np.array([755., 656.], dtype=np.float32), np.array([319., 347.], dtype=np.float32)), (np.array([758., 685.], dtype=np.float32), np.array([340., 370.], dtype=np.float32)), (np.array([395., 428.], dtype=np.float32), np.array([ 17., 118.], dtype=np.float32)), (np.array([595., 463.], dtype=np.float32), np.array([167., 141.], dtype=np.float32)), (np.array([798., 667.], dtype=np.float32), np.array([355., 346.], dtype=np.float32))]
    

    kjeller_fly_medium = cv2.imread("dataset/map/kjeller_kart_ny.png")
    kjeller_kart_lite = cv2.imread("dataset/map/kjeller_flyfoto_ny.png")

    kjeller_fly_medium = cv2.cvtColor(kjeller_fly_medium, cv2.COLOR_BGR2RGB)
    kjeller_kart_lite = cv2.cvtColor(kjeller_kart_lite, cv2.COLOR_BGR2RGB)

    match_points = gkm.get_matchpoints(kjeller_fly_medium, kjeller_kart_lite, "all")

    overlay = merge_images(kjeller_fly_medium, kjeller_kart_lite, match_points)
    plt.imsave("overlay_homography_all_points.png", overlay, dpi=400.)
    
def main():
    plot_matches = True

    kjeller_fly_medium = cv2.imread("dataset/map/kjeller_kart_ny.png")
    kjeller_kart_lite = cv2.imread("dataset/map/kjeller_flyfoto_ny.png")

    kjeller_fly_medium = cv2.cvtColor(kjeller_fly_medium, cv2.COLOR_BGR2RGB)
    kjeller_kart_lite = cv2.cvtColor(kjeller_kart_lite, cv2.COLOR_BGR2RGB)

    match_points = gkm.get_matchpoints(kjeller_fly_medium, kjeller_kart_lite, "all")

    if plot_matches:
        colours = ["*b", "*g", "*r", "*m", "*c", "*y"]
        plt.subplot(121)
        plt.suptitle("Some of the Matches")
        plt.imshow(kjeller_fly_medium)
        plt.title('kjeller_kart')
        for i in range(len(match_points)):
            if i%13 == 0:
                plt.plot(match_points[i][0][0], match_points[i][0][1], colours[i%6])
        plt.subplot(122)
        plt.imshow(kjeller_kart_lite)
        plt.title('kjeller_flyfoto')
        for i in range(len(match_points)):
            if i%13 == 0:
                plt.plot(match_points[i][1][0], match_points[i][1][1], colours[i%6])
        plt.show()

    print("Using RANSAC")
    overlay = merge_images(kjeller_fly_medium, kjeller_kart_lite, match_points)
    plt.imshow(overlay)
    plt.title("Overlapping area")
    plt.show()
    plt.imsave("overlay_homography_all_points.png", overlay, dpi=400.)

if __name__ == "__main__":
    # main6()
    # main1()
    # main3()
    # main4()
    # main5()
    main()
    