
#=========================================================
# Create Disparity map from Stereo Vision
#=========================================================

import cv2
import matplotlib.pyplot as plt
# For each pixel algorithm will find the best disparity from 0
# Larger block size implies smoother, though less accurate disparity map

# Set disparity parameters
# Note: disparity range is tuned according to specific parameters obtained through trial and error. 
imgL = cv2.imread('left1.png')


imgR = cv2.imread('right1.png') 

imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
#imgLgray = cv2.resize(imgLgray, (1200, 1200), interpolation = cv2.INTER_AREA)

block_size = 3
min_disp = -1
max_disp = 31
num_disp = max_disp - min_disp

# Create Block matching object. 
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
	numDisparities = num_disp,
	blockSize = block_size,
	uniquenessRatio = 5,
	speckleWindowSize = 5,
	speckleRange = 2,
	disp12MaxDiff = 2) 


#stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=win_size)

# Compute disparity map
disparity_map = stereo.compute(imgLgray, imgRgray)

plt.imshow(disparity_map,'gray')
plt.show()

