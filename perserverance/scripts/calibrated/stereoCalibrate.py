#!/usr/bin/python3
##Reference

#[1] https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
#[2]https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from params import *

class calibration:
    def __init__(self):
        self.imageL = cv.imread("ZL0_0050_0671382043_081ECM_N0031950ZCAM08013_063085J01.png")
        self.imageR = cv.imread("ZR0_0050_0671382043_081ECM_N0031950ZCAM08013_063085J01.png")

        cv.imshow("left", self.imageL)
        cv.imshow("right", self.imageR)
        cv.waitKey(0)

        xmls = list()
        for files in os.listdir():
            if files.endswith('.xml'):
                xmls.append(files)

        pixelSize =  0.0074#size in mm
        shapeL = self.imageL.shape[:-1]
        shapeR = self.imageL.shape[:-1]
        params = Params(xmls, pixelSize, shapeL)



        ########## Stereo Rectification #################################################

        rectifyScale= 0
        rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(params.cameraMatrixL, params.distL,
                params.cameraMatrixR, params.distR, shapeR, params.extrinscis['R'], params.extrinscis['T'], rectifyScale,(0,0))
        stereoMapL = cv.initUndistortRectifyMap(params.cameraMatrixL, params.distL, rectL, projMatrixL, shapeL, cv.CV_16SC2)
        stereoMapR = cv.initUndistortRectifyMap(params.cameraMatrixR, params.distR, rectR, projMatrixR, shapeR, cv.CV_16SC2)

        imgL = cv.remap(self.imageL, stereoMapL[0], stereoMapL[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
        imgR = cv.remap(self.imageR, stereoMapR[0], stereoMapR[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

        cv.imwrite("UndistortR.png", imgR) 
        cv.imwrite("UndistortL.png", imgL)
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
        create_point_cloud_file(output_points, output_colors, 'point8.ply')



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

if __name__ == '__main__':
    calibration()
