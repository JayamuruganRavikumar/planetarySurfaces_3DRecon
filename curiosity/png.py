import os
import numpy as np
import gdal 
import matplotlib.pyplot as plt 

def main():
    dir = os.getcwd()
    img_quality = 1000
    img_list = []
    for file in os.listdir(path = dir):
        if file.endswith(".IMG"):
            img_list.append(file)
            print(file)
    # files above ~1MB only
    good_imgs = []
    for file in img_list:
        img_size = os.path.getsize(os.path.join(dir,file))
        if img_size > img_quality:
            good_imgs.append(file.replace(".IMG",".LBL"))
    #Giving name to the file
    file_name = []
    band_dim = []
    imgs = []
    for file in good_imgs:
        file_name.append(file.split("LBL"))
    print(file_name)

    for file in good_imgs:
        img = gdal.Open(os.path.join(dir,file))
        #print(f'image: {file}')

        img_bands = []
        for band in range(img.RasterCount):
            band += 1
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
 
    i = 0
    for img_name in file_name:
        
        newName = os.path.join(dir,img_name[0]+'png')
        plt.imsave(newName, imgs[i])
        #gdal.Translate(newName, good_imgs[i])
        i += 1

if __name__ == "__main__":
    main()

