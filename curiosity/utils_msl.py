from osgeo import gdal
import matplotlib.pyplot as plt
import sys
import numpy as np
import requests
import os
import pvl  # to interpret the .LBL file
import cv2

# verbose print
DEBUG = 1
LOCAL_DIR = "~/wise/project/curiosity/19"
IMG_SIZE_THRESHOLD = 1000000 # ~1MB


def get_img_from_NASA(camera='navcam'):
    """
    Gets an example image of MSL's mastcam/navcam directly from NASA's server directly
  
    *For testing only* 

    Parameters:
    camera (str): 'navcam' (Default) or 'mastcam'
  
    Returns:
    dir (str): image directory is the root folder
    good_imgs (list): one test image in the directory (LBL file is enough) 
    
    Maintainer: Tushar Jayesh Barot
    """

    lbl = None  
    if camera == 'navcam':
        img = "NLA_400163518EDR_F0040000NCAM00105M1.IMG"
        lbl = "NLA_400163518EDR_F0040000NCAM00105M1.LBL"

        url = "https://pds-imaging.jpl.nasa.gov/data/msl/MSLNAV_0XXX/DATA/SOL00030/"
    elif camera == 'mastcam':
        img = "0019MR0000530080100146C00_DRCL.IMG"
        lbl = "0019MR0000530080100146C00_DRCL.LBL"
        url = "https://pds-imaging.jpl.nasa.gov/data/msl/MSLMST_0001/DATA/RDR/SURFACE/0019/"
    else:
        return None

    # saving .IMG file
    r = requests.get(url+img)
    open(img, 'wb').write(r.content)
    
    # saving .LBL file
    if lbl != None:
        r = requests.get(url+lbl)
        open(lbl, 'wb').write(r.content)

    dir = os.getcwd()

    return dir, [lbl]

def find_img_local(dir=LOCAL_DIR, img_quality=IMG_SIZE_THRESHOLD):
    """
    Get list of image filenames from local directory which are above a certain size (~1MB)
  
    Parameters:
    dir (str): local directory of the images
    img_quality (int): minimum size of images in bytes
  
    Returns:
    dir (str): image directory
    good_imgs (list): .LBL filename of good images in the directory 
                (gdal takes .LBL files as input for both cam)
    
    Maintainer: Tushar Jayesh Barot
    """

    # read all files into a list
    print(f"Local directory of .IMG files: {dir}")

    # filter .img files only
    img_list = []
    for file in os.listdir(path = dir):
        if file.endswith(".IMG"):
            img_list.append(file)

    # files above ~1MB only
    good_imgs = []
    for file in img_list:
        img_size = os.path.getsize(os.path.join(dir,file))
        if img_size > img_quality:
            good_imgs.append(file.replace(".IMG",".LBL"))

    return dir, good_imgs

def gdal_msl(dir,img_filenames):
    """
    Open .LBL and .IMG files as 1D/3D numpy arrays using gdal package for MSL mission

    Parameters:
    dir (str): local directory of the images (.LBL and .IMG file)
    img_filenames (list): file name of images that have to be opened
  
    Returns:
    imgs (list): list of images as numpy arrays  
    img_bands (list): number of bands for each corresponding image in `imgs` (1 for grayscale, 3 for RGB)

    Maintainer: Tushar Jayesh Barot
    """
    imgs = []
    band_dim = []
    # view all color channels of each good image
    for file in img_filenames:
        img = gdal.Open(os.path.join(dir,file))
        print(f'image: {file}')

        img_bands = []
        for band in range(img.RasterCount):
            band += 1
            if(DEBUG):
                print ("[ GETTING BAND ]: ", band)
            np_band = np.array(img.GetRasterBand(band).ReadAsArray())
            img_bands.append(np_band)        
        
        img_x = img_bands[0].shape[0]
        img_y = img_bands[0].shape[1]
        dim = len(img_bands)

        if dim == 1:
            # imgs = np.zeros((img_x,img_y), 'uint8')
            imgs.append(img_bands[0])
            band_dim.append(dim)
        else:
            img_rgb = np.zeros((img_x,img_y,dim), 'uint8')
            # Combining all color channels into a single array
            for i in range(dim):
                img_rgb[:,:, i] = img_bands[i]

            imgs.append(img_rgb)
            band_dim.append(dim)
            return imgs, band_dim
    return imgs, band_dim


def viewimage_msl(img, band_dim):
    """
    view 1D/3D image (1 or 3 band image) 

    Parameters:
    img (np_array): image as a 1D/3D np_array
    band_dim (int): dimension of bands in image (grayscale or color)
  
    Returns:
    

    Maintainer: Tushar Jayesh Barot
    """

    print(f'Viewing image of size: {img.shape} : {band_dim} band(s)')
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    
    if band_dim == 1:
        c = ax.imshow(img,cmap='gray')
    else:
        c = ax.imshow(img)

    plt.title('MSL Image',fontweight ="bold")
    plt.show()


def readLBL_msl(dir, lbl_filenames):
    """
    Open .LBL as a dictionary using pvl package for MSL mission

    Parameters:
    dir (str): local directory of the images 
    lbl_filenames (list): file name of .LBL of images that have to be opened (.LBL file)
  
    Returns:
    lbl_metadata (list): dictionary of metadata read from the .LBL file

    Maintainer: Tushar Jayesh Barot
    """

    lbl_metadata = []
    for file in lbl_filenames:
        lbl_metadata.append(pvl.load(os.path.join(dir,file)))
        
    return lbl_metadata


def cameraname_msl(lbl_meta):
    """
    fetch name of the camera/instrument

    Name of instrument is from the corresponding metadata dict passed as argument.
    The metadata dict is read using pvl package from the .LBL file

    Parameters:
    lbl_meta (dict): metadata dict read from .LBL file

    Returns:
    camera_name (str): name of the camera/instrument

    Maintainer: Tushar Jayesh Barot
    """

    camera_name = lbl_meta['INSTRUMENT_ID']
    if(DEBUG):
        print(f'Camera: {camera_name}')
    return camera_name

def savePNG_msl(dir, img_name, img):
    """
    saves numpy array as image

    Parameters:
    dir (str): destination directory for the .png image
    img_name (str): Example: 'img1'. The extension '.png' will be added  
    img (np_array): image  as np array

    Returns:

    Maintainer: Tushar Jayesh Barot
    """


    full_filename = os.path.join(dir,img_name+'.png')
    plt.imsave(full_filename, img)



if __name__ == "__main__":
    """
    Testing the util functions here
    """
    
#    if  input("Do you want to download a test file? [y/n]  ----> ").lower() == 'y':
#        print("Downloading test image. This may take some time")
#        dir,lbl_filenames = get_img_from_NASA('mastcam')
#        print(f"Downloaded test image {lbl_filenames} at {dir}")
#    else:
    dir,lbl_filenames = find_img_local()
#        
    img, band_dim = gdal_msl(dir,lbl_filenames)
#
    # testing viewimage_msl
#    for i in range(len(img)):
#        viewimage_msl(img[i], band_dim[i])
#        
#    # testing readLBL_msl for 3 images
#    for i in range(len(img)):
#        if i < 3:
#            print(f"reading metadata image: {i+1}")
#            lbl_meta = readLBL_msl(dir, lbl_filenames)
#
#    # testing cameraname_msl
#    cameraname_msl(lbl_meta[0])
#
    # testing savePNG_msl
    savePNG_msl(dir=os.getcwd(), img_name='test',img=img[0])
