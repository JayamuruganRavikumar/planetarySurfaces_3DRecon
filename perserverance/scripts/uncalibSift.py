#!/usr/bin/python3

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

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


# Input images

image_left = cv.imread("lefrncam_57.png", cv.IMREAD_GRAYSCALE)
image_right = cv.imread("rightrncam_57.png", cv.IMREAD_GRAYSCALE)

# Keypoints ans descriptors

keyp1, keyp2, desc1,desc2, flann_matches = keypoints_and_desscriptors_sift(image_left, image_right)
good_matches, mask, good_point1, good_point2 = lowes_test(flann_matches, 1, keyp1, keyp2)
draw_matches(image_left, image_right, keyp1, keyp2, mask, flann_matches)

f ,inliners, pts1, pts2 = fundamental_matrix(good_point1, good_point2)

# Rectification

h1, w1 = image_left.shape
h2, w2 = image_right.shape
_, H1, H2 = cv.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts2), f, imgSize=(w1, h1)
)

imgl_rectified = cv.warpPerspective(image_left, H1, (w1, h1))
imgr_rectified = cv.warpPerspective(image_right, H2, (w2, h2))
cv.imwrite("rectified_1.png", imgl_rectified)
cv.imwrite("rectified_2.png", imgr_rectified)

min_disp = -1
max_disp = 31
block_size = 10
num_disp = max_disp - min_disp  # Needs to be divisible by 16
stereo = cv.StereoSGBM_create(minDisparity= min_disp,
    numDisparities = num_disp,
    blockSize = block_size,
    uniquenessRatio = 5,
    speckleWindowSize = 3,
    speckleRange = 2,
    disp12MaxDiff = 2) 

disparity_SGBM = stereo.compute(imgl_rectified, imgr_rectified)
plt.imshow(disparity_SGBM, "gray")
plt.colorbar()
plt.show()
