#!/usr/bin/python3
##Reference

#[1] https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def create_point_cloud_file(vertices, colors, filename):
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

imageL = cv.imread("ZL0_0050_0671382043_081ECM_N0031950ZCAM08013_063085J01.png")
imageR = cv.imread("ZR0_0050_0671382043_081ECM_N0031950ZCAM08013_063085J01.png")

shape = imageL.shape[:-1]
#Camera matrix = 
#\begin{bmatrix}
#f_x & 0   & c_x \\
#0   & f_y & c_y \\
#0   & 0   & 1
#\end{bmatrix}

cameraMatrixL =  np.float64([[8596.278146385768596, 0, 6.000005421983138e+02], [0, 8.596278146385757e+03, 8.240006339949988e+02], [0, 0, 1]])
cameraMatrixR = np.float64([[8.591069853194515e+03, 0, 6.000005934884033e+02], [0, 8.591069853194515e+03, 8.240005217915393e+02], [0, 0, 1]])

#distortoin [k1, k2, p1, p2, k3]
distL = np.float64([[-3.251430000000000e-04, -86.609691253161440, 0, 0, -1.161090153877255e+05]])
#distL = np.float64([[0,0,0,0,0]])
distR = np.float64([[-5.833330000000000e-05, -85.784174300182290, 0, 0, -6.837455653139655e+04]])
#distR = np.float64([[0,0,0,0,0]])

#Rotation
R = np.float64([[0.999263714429505,	-0.0371848860812231, -0.00945056991269422], [0.0370307852381919, 0.999186205201875,	-0.0159890049383676], [0.0100374284152453, 0.0156272704399294, 0.999827504347328]])
#Translation

trans = np.float64([[0.144973000000000, 0.196717000000000, 0.000979999999999981]])
trans = np.transpose(trans)

##If images are downsampled
#heightL, widthL, channelsL = imgL.shape
#newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
#
#heightR, widthR, channelsR = imgR.shape
#newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))


########## Stereo Rectification #################################################

rectifyScale= 0
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(cameraMatrixL, distL, cameraMatrixR, distR, shape, R, trans, rectifyScale,(0,0))
print(Q)
stereoMapL = cv.initUndistortRectifyMap(cameraMatrixL, distL, rectL, projMatrixL, shape, cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(cameraMatrixR, distR, rectR, projMatrixR, shape, cv.CV_16SC2)

stereoMapL_x = stereoMapL[0]
stereoMapL_y = stereoMapL[1]
stereoMapR_x = stereoMapR[0]
stereoMapR_y = stereoMapR[1]

imgL = cv.remap(imageL, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
imgR = cv.remap(imageR, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

cv.imshow("Undistort R", imgR) 
cv.imshow("Undistort L", imgL)
cv.waitKey(0)


grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)


# Set disparity parameters
# Note: disparity range is tuned according to specific parameters obtained through trial and error. 
block_size = 15
min_disp = -1
max_disp = 31
num_disp = max_disp - min_disp # Needs to be divisible by 15

# Create Block matching object. 
stereo = cv.StereoSGBM_create(minDisparity= min_disp,
	numDisparities = num_disp,
	blockSize = block_size,
	uniquenessRatio = 5,
	speckleWindowSize = 5,
	speckleRange = 2,
	disp12MaxDiff = 2,
	P1 = 8 * 3 * block_size**2,#8*img_channels*block_size**2,
	P2 = 32 * 3 * block_size**2) #32*img_channels*block_size**2)


#stereo = cv2.StereoBM_create(numDisparities=num_disp, blockSize=win_size)

# Compute disparity map
disparity_map = stereo.compute(grayL, grayR)

# Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
plt.imshow(disparity_map,'gray')
plt.show()


#print(disparity_map.dtype)
disparity_map = np.float32(np.divide(disparity_map, 16.0))
#print(disparity_map.dtype)

# Reproject points into 3D
points_3D = cv.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
# Get color of the reprojected points
colors = cv.cvtColor(imgR, cv.COLOR_BGR2RGB)

# Get rid of points with value 0 (no depth)
mask_map = disparity_map > disparity_map.min()

# Mask colors and points. 
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

create_point_cloud_file(points_3D, colors, 'point8.ply')
