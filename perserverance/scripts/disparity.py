
'''
Create disparity map and projec it to 3d points
'''
#=========================================================
# Create Disparity map from Stereo Vision
#=========================================================
from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import numpy as np

#============Function to create a point cloud==============
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

def write_ply(filename, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(filename, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

#================================================================

def main():
    # For each pixel algorithm will find the best disparity from 0
    # Larger block size implies smoother, though less accurate disparity map

    imgL = cv2.imread('lefrncam_57.png')
    imgR = cv2.imread('rightrncam_57.png') 

    imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    #resize the image (works when both the images are of the smae size)
    #imgLgray = cv2.resize(imgLgray, (1200, 1200), interpolation = cv2.INTER_AREA)

    # Set disparity parameters
    # Note: disparity range is tuned according to specific parameters obtained through trial and error. 
    #Use the calibration python file for knowing the parameters
    #block size should be odd
    #Number of Disparities should be divisible by 16

    block_size = 5
    min_disp = 0
    max_disp = 16
    num_disp = max_disp - min_disp

    # Create Block matching object. 
    stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
        numDisparities = num_disp,
        blockSize = block_size,
        uniquenessRatio = 5,
        speckleWindowSize = 8,
        speckleRange = 2,
        disp12MaxDiff = 2) 


    # Compute disparity map
    print('Computing disparity')
    disparity_map = stereo.compute(imgLgray, imgRgray)

    plt.imshow(disparity_map,'gray')
    plt.show()

    #========================================================
    # Disparity to 3d Points
    #========================================================
    #Q matrix for nav cam
    #Q - reprojection matrix to get 3d points from disparity_map 
    cv_file = cv2.FileStorage()
    cv_file.open('stereo.xml', cv2.FileStorage_READ)
    Q = cv_file.getNode('q').mat()
    #cx =   #coordinates of the principal point of the leftcam
    #cy =
    #cxr =  #coordinates of the principal point of the rightcam 
    #f =    #focal length 
    #Tx =   #baseline lenght

    #Q = np.float32([[1, 0, 0, -cx],[0,-1, 0,  -cy],[0, 0, 0, -f],[0, 0, -1/Tx, (cx - cxr)/Tx]])
    #Q matrix for mastcam

    disparity_map = np.float32(np.divide(disparity_map, 16.0))
    reconstruct = cv2.reprojectImageTo3D(disparity_map, Q, handleMissingValues=False)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disparity_map > disparity_map.min()
    out_points = reconstruct[mask]
    out_colors = colors[mask]
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')

if __name__ == "__main__":
    main()
