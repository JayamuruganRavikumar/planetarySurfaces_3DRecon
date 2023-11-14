#!/usr/bin/python3
##Reference:
#
#[1]https://stackoverflow.com/questions/51197091/how-does-the-lowes-ratio-test-work
#[2]https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
#[3]https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system
#[4]https://docs.opencv.org/4.2.0/d4/d5d/group__features2d__draw.html
#[5]https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html
#[6]https://en.wikipedia.org/wiki/Eight-point_algorithm#The_normalized_eight-point_algorithm
#[7] Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
#[8]https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
#[9]https://docs.opencv.org/3.4/db/d27/tutorial_py_table_of_contents_feature2d.html
#

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#in mm
focal_length = 19.1
sensor_size_x = 0.0064
sensor_size_y = 0.0064
sensor_size =  sensor_size_x*sensor_size_y


#Get the KeyPoints
def keypoints_and_desscriptors_sift(image_left, image_right):
    """Using SIFT(Scale invariant feature transform) and FLANN matcher to 
    obtain the keypoints and the descriptors for the stero pair[2]
    Input : left and right images
    Output: keypoints 1, keypoints 2, descriptors1, descriptors2, flann_matches
    """
    sift = cv.SIFT_create()
    key1, desc1 = sift.detectAndCompute(image_left, None)
    key2, desc2 = sift.detectAndCompute(image_right, None)

    #FLANN Enhanced Nearest Neighbour Method
    # The keypoints of the first image is matched with the second one
    # i.e the left with the right image. 
    # The k = 2 keeps the best 2 matches for each point with smallest
    # distance
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params,search_params)
    Flann_matches = flann.knnMatch(desc1,desc2,k=2)

    return key1, key2, desc1, desc2, Flann_matches

def keypoints_and_desscriptors_ORB(image_left, image_right):
    """Using ORB to extract the features
    Input : left and right images
    Output: keypoints 1, keypoints 2, descriptors1, descriptors2, bf_mathces
    """

    orb = cv.ORB_create(nfeatures=10000)
    # find the keypoints and descriptors with ORB
    key1, des1 = orb.detectAndCompute(image_left,None)
    key2, des2 = orb.detectAndCompute(image_right,None)


    #:________________________Brute Force Matcher_____________________________
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match descriptors._
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # print(len(matches))
    
    # Select first 30 matches.
    bf_mathces = matches[:50]


    return key1, key2, des1, des2, bf_mathces

def lowes_test(matches, threshold, kp1, kp2):
    """ lowes test to keep the good points for caluculating distinctive image
    features[7][1]
    Input : flann_matches, threshold, keypoints1, keypoints2
    Output: Gives the good matches, matchesMask
    """
    matchesMask = [[0, 0] for i in range(len(matches))]
    filtered_matches = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < threshold*n.distance:
            # Keep this keypoint pair
            matchesMask[i] = [1, 0]
            filtered_matches.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    return filtered_matches, matchesMask, pts1, pts2

def draw_matches(image_left, image_right, key1, key2, matchesMask, flann_matches):
    """Matches between the images[4][5]
    Input : left image, right image, keypoints1, keypoints2, matchesMask, flann matches
    """

    params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask[200:300],
                       flags=cv.DrawMatchesFlags_DEFAULT)

    image = cv.drawMatchesKnn(
            image_left, 
            key1,
            image_right,
            key2,
            flann_matches[200:300],
            None,
            **params)
    cv.imshow("Matched points", image)
    cv.waitKey(0)

def fundamental_matrix(kp1, kp2):
    """Using the good matchs to get a good estimate of the fundamental matrix[6]
    Input : keypoints1, keypoints2 (of good matches)
    Output: f matrix, inliers, keypoints1, keypoints2
    """
    pts1 = np.int32(kp1)
    pts2 = np.int32(kp2)
    fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

    # We select only inlier points
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]

    return fundamental_matrix, inliers, pts1, pts2

def K_matrix(image):
    x_pixels = len(image[0])
    y_pixels = len(image)

    pixel_size_x = x_pixels / sensor_size_x   #in pixels/mm
    pixel_size_y = y_pixels / sensor_size_y     #in pixels/mm

    f_x = focal_length * pixel_size_x     #pixels
    f_y = focal_length * pixel_size_y     #pixels
    global f_pix
    

    f_pix = focal_length * ((pixel_size_x * pixel_size_y) / 2)
    K = np.asarray([[f_x, 0, x_pixels/2],
                    [0, f_y, y_pixels/2],
                    [0, 0, 1]])
    return K


def E_matrix(F,image_left, image_right):
    #E = Kt * F * K
    K_l = K_matrix(image_left)
    K_r = K_matrix(image_right)
    E = np.dot(F, K_r)
    E = np.dot(np.transpose(K_l), E)

    return E

def Rot_Tran_Matrix(E):

    U,S,V = np.linalg.svd(E)

    V = V.T
    mid = np.float32([ [0,-1,0],
                      [1, 0,0],
                      [0, 0,1] ])

    prodA = np.dot(mid, V.T)

    rotation = np.dot(U, prodA)

    translation = np.transpose(np.matrix([U[0][2],U[1][2],U[2][2]]))

    return rotation, translation

def compute_P(K, R, T):
    #calculate intrinsics matrix
    M_int = K
    M_int = np.c_[M_int, [0, 0, 0]]

    #calculate extrinsics
    M_ext = R
    M_ext = np.c_[M_ext, T]
    M_ext = np.r_[M_ext, [[0, 0, 0, 1]]]

    #calculate projection
    P = np.dot(M_int, M_ext)

    return P

def disparity_to_depth(baseline, f, img):
    """This is used to compute the depth values from the disparity map
    Input : baseline, fundamental matrix, disparity"""

    # Assumption image intensities are disparity values (x-x') 
    depth_map = np.zeros((img.shape[0], img.shape[1]))
    depth_array = np.zeros((img.shape[0], img.shape[1]))

    for i in range(depth_map.shape[0]):
        for j in range(depth_map.shape[1]):
            depth_map[i][j] = 1/img[i][j]
            depth_array[i][j] = baseline*f/img[i][j]

    return depth_map, depth_array

def create_point_cloud_file(vertices, colors, filename):
    """Creating a pont cloud
    Input vertices, colors, filename"""

    colors = colors.reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3),colors])

    ply_header = '''ply
        format ascii 1.0
        element vertex %(vert_num)d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        end_header
        '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num=len(vertices)))
        np.savetxt(f,vertices,'%f %f %f %d %d %d')


def drawlines(image_left, image_right, lines, pts1src, pts2src):
    """img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines""" 
    r, c = image_left.shape
    imglcolor = cv.cvtColor(image_left, cv.COLOR_GRAY2BGR)
    imgrcolor = cv.cvtColor(image_right, cv.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        imglcolor = cv.line(imglcolor, (x0, y0), (x1, y1), color, 1)
        imglcolor = cv.circle(irg1color, tuple(pt1), 5, color, -1)
        imgrcolor = cv.circle(imgrcolor, tuple(pt2), 5, color, -1)
    return imglcolor, imgrcolor

def plot_3D(im1, im2, img3D):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for i in range(0,len(im1), 1):
        for j in range(0,len(im1[0]), 1):
            x = img3D[j][i][0]
            y = img3D[j][i][1]
            z = img3D[j][i][2]
            '''
            #print '\ni:', i
            #print '\nj:', j
            #print '\nlen(im1):', len(im2)
            #print '\nlen(im1[0]):', len(im2[0])
            #'''
            ax.scatter(x,y,z, c=im2[i,j], marker='o')

    ax.set_xlabel('x axis')
    ax.set_xlabel('y axis')
    ax.set_xlabel('z axis')


def ply_from_list(le_list):

    list_len = len(le_list)

    le_file = open('my_sparse_ply.ply', 'w')

    #go through and find all the points that arent infinity
    inf_ct = 0
    for i in range(list_len):
        for j in range(len(le_list[0, 0])):
            a = str(le_list[i, 0, j])
            b = str(le_list[i, 0, j])
            c = str(le_list[i, 0, j])

            if (a and b and c) != 'inf':
                if (a and b and c) != '-inf':
                    inf_ct += 1


    le_file.write(( 'ply\n' +
                    'format ascii 1.0\n' +
                    'element vertex ' + str(inf_ct) + '\n'
                    'property float x\n' +
                    'property float y\n' +
                    'property float z\n' +
                    'end_header\n'))

    for i in range(list_len):
        for j in range(len(le_list[0, 0])):
            a = str(le_list[i, 0, j])
            b = str(le_list[i, 0, j])
            c = str(le_list[i, 0, j])

            if (a and b and c) != 'inf':
                if (a and b and c) != '-inf':
                    le_file.write(a + ' ' + b + ' ' + c + '\n')

    le_file.close()
##________________________________Input Images________________________________##


image_left = cv.imread("lefrncam_57.png", cv.IMREAD_GRAYSCALE)
image_right = cv.imread("rightrncam_57.png", cv.IMREAD_GRAYSCALE)

##_______________________________Keypoints and descriptors___________________##

keyp1, keyp2, desc1,desc2, flann_matches = keypoints_and_desscriptors_sift(image_left, image_right)
good_matches, mask, good_point1, good_point2 = lowes_test(flann_matches, 0.7, keyp1, keyp2)
draw_matches(image_left, image_right, keyp1, keyp2, mask, flann_matches)

f ,inliners, pts1, pts2 = fundamental_matrix(good_point1, good_point2)

#E = E_matrix(f, image_left, image_right)
#R, T = Rot_Tran_Matrix(E)
K = K_matrix(image_left)

##_________________________________Epilines__________________________________##


#lines1 = cv.computeCorrespondEpilines(
#    pts2.reshape(-1, 1, 2), 2, f)
#lines1 = lines1.reshape(-1, 3)
#img5, img6 = drawlines(image_left, image_right, lines1, pts1, pts2)
#
## Find epilines corresponding to points in left image (first image) and
## drawing its lines on right image
#lines2 = cv.computeCorrespondEpilines(
#    pts1.reshape(-1, 1, 2), 1, f)
#lines2 = lines2.reshape(-1, 3)
#img3, img4 = drawlines(image_left, image_right, lines2, pts2, pts1)
#
#plt.subplot(121), plt.imshow(img5)
#plt.subplot(122), plt.imshow(img3)
#plt.suptitle("Epilines in both images")
#plt.show()
#
##___________________________________rectification/Undistort___________________________##


h1, w1 = image_left.shape
h2, w2 = image_right.shape
_, H1, H2 = cv.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts2), f, imgSize=(w1, h1)
)

imgl_rectified = cv.warpPerspective(image_left, H1, (w1, h1))
imgr_rectified = cv.warpPerspective(image_right, H2, (w2, h2))
#cv.imwrite("rectified_1.png", imgl_rectified)
#cv.imwrite("rectified_2.png", imgr_rectified)

fig, axes = plt.subplots(1, 2, figsize=(15, 10))
axes[0].imshow(imgl_rectified, cmap="gray")
axes[1].imshow(imgr_rectified, cmap="gray")
axes[0].axhline(250)
axes[1].axhline(250)
axes[0].axhline(450)
axes[1].axhline(450)
plt.suptitle("Rectified images")
#plt.savefig("rectified_images.png")
plt.show()

##________________________________Disparity map________________________________##

#_________________Using StereoBM
stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
disparity_BM = stereo.compute(imgl_rectified, imgr_rectified)
plt.imshow(disparity_BM, "gray")
plt.colorbar()
plt.show()

disparity_map = np.float32(np.divide(disparity_BM, 16.0))

#_____________ Using StereoSGBM
# Set disparity parameters. Note: disparity range is tuned according to
##  specific parameters obtained through trial and error.
#min_disp = -1
#max_disp = 31
#block_size = 5
#num_disp = max_disp - min_disp  # Needs to be divisible by 16
#stereo = cv.StereoSGBM_create(minDisparity= min_disp,
#    numDisparities = num_disp,
#    blockSize = block_size,
#    uniquenessRatio = 5,
#    speckleWindowSize = 3,
#    speckleRange = 2,
#    disp12MaxDiff = 2) 
#
#disparity_SGBM = stereo.compute(imgl_rectified, imgr_rectified)
#plt.imshow(disparity_SGBM, "gray")
#plt.colorbar()
#plt.show()

#disparityF = disparity_SGBM.astype(float)
#maxv = np.max(disparityF.flatten())
#minv = np.min(disparityF.flatten())
#disparityF = 255.0*(disparityF-minv)/(maxv-minv)
#disparityU = disparityF.astype(np.uint8)

#disparity_map = np.float32(np.divide(disparity_SGBM, 16.0))

#P = compute_P(K, R, T)
#
#hom_pts = cv.triangulatePoints(P, P, pts1.T, pts2.T)
#
#cld = cv.convertPointsFromHomogeneous(hom_pts.T)
#
#ply_from_list(cld)

#Reprojection matrix
cx = len(image_left[0]) / 2
cy = len(image_left) / 2
Tx = 0.422 
# projection matrix from opencv docs
Q = np.float32([[1, 0, 0, -cx],
                [0, 1, 0, cy],
                [0, 0, 0, -f_pix],
                [0, 0, 1, 0]])

#From https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0
#Q = np.float32([[1, 0, 0, 0],
#                [0, -1, 0, 0],
#                [0, 0, 0, focal_length*0.5],
#                [0, 0, 1, 0]])
#Reproject to 3d
image3d = cv.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)

colors = cv.cvtColor(image_left, cv.COLOR_BGR2RGB)
mask = disparity_map > disparity_map.min()


create_point_cloud_file(image3d, colors, 'test2.ply')
