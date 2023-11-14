#!/usr/bin/python3
'''
Rover Reconstruction Group
M2020 Utility Script
Author: Ashwin Nedungadi
Maintainer: ashwin.nedungadi@tu-dortmund.de
Licence: The MIT License
Status: Production
'''

import gdal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os, shutil
from pathlib import Path
import cv2
import xml.etree.ElementTree as etree

def normalize(input):

    """ Given an image numpy array, normailzes the image to Uint8. """
    # Double check if downsampling is happening here, plt only accepts Uint8 and not Uint16 cv2.CV_16U
    normalized_image = cv2.normalize(input, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Manual Way of Normalizing
    #max_val = input.max()
    #min_val = 0
    #k = 255/max_val
    #normalized_image = input*k

    return normalized_image

def get_imgs(input_directory):
    """ A function that would get images from PDS Server into the local directory, ignores thumbnails """
    # read all files into a list

    os.chdir(input_directory)
    print(f"Local directory of .IMG files: {os.getcwd()}")

    # filter .img files only
    img_list = []
    for file in os.listdir(path=input_directory):
        if file.endswith(".IMG") or file.endswith(".png"):
            img_list.append(file)

    # files above ~100KB only
    imgs = []
    for file in img_list:
        img_size = os.path.getsize(os.path.join(input_directory, file))
        if img_size > 100000:
            imgs.append(file)
    return imgs

def read_xml(input_xml):
    """ reads all xml tags for a particular image and returns as dictionary """
    #return gdal.Info(input_image, format='json')
    # Question: is it better to accept an xml filepath directly or an image filepath and then find the corresponding xml?
    tree = etree.parse(input_xml)

    for elem in tree.iter():
        tag = elem.tag
        new_tag = tag.split('}')
        new_tag = new_tag.pop()

        if new_tag == 'alternate_id':
            print("Filename: ", elem.text)

        elif new_tag == 'start_date_time':
            print("start_date_time: ", elem.text)
        elif new_tag == 'stop_date_time':
            print("stop_date_time: ", elem.text)
        elif new_tag == 'processing_level':
            print("processing_level: ", elem.text)

        elif new_tag == 'ops_instrument_key':
            print("ops_instrument_key: ", elem.text)
        elif new_tag == 'sol_number':
            print("sol_number: ", elem.text)
        elif new_tag == 'product_type_name':
            print("product_type_name: ", elem.text)

        elif new_tag == 'processing_algorithm':
            print(new_tag, ": ", elem.text)
        elif new_tag == 'color_filter_array_type':
            print(new_tag, ": ", elem.text)

def read_img(file):
    """ reads .IMG file and returns the rgb array """
    img_bands = []
    band_dim = []
    image_array = []

    image = gdal.Open(os.path.join(os.getcwd(), file))
    #print(f'image: {file}')
    for band in range(image.RasterCount):
        band += 1

        np_band = np.array(image.GetRasterBand(band).ReadAsArray())
        img_bands.append(np_band)

    img_x = img_bands[0].shape[0]
    img_y = img_bands[0].shape[1]
    dim = len(img_bands)

    if dim == 1:
        # imgs = np.zeros((img_x,img_y), 'uint8')
        image_array.append(img_bands[0])
        band_dim.append(dim)

    else:
        img_rgb = np.zeros((img_x, img_y, dim), 'uint16')
        # Combining all color channels into a single array
        for i in range(dim):
            img_rgb[:, :, i] = img_bands[i]

        image_array.append(img_rgb)
        band_dim.append(dim)
        return image_array[0], band_dim[0]

    return image_array[0], band_dim[0]

def view(input_image):
    """ views an .img or .png file when called with the filename as argument. """

    if input_image.endswith('.IMG'):

        image, bands = read_img(input_image)
        image = normalize(image)

        if bands == 1:
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111)

            c = ax.imshow(image, cmap='gray')

            plt.title('M2020 BW Image', fontweight="bold")
            plt.show()
        else:
            fig = plt.figure(figsize=(15, 15))
            ax = fig.add_subplot(111)

            c = ax.imshow(image)

            plt.title('M2020 RGB Image', fontweight="bold")
            plt.show()

    elif input_image.endswith('.png'):
        # To implement png viewing part here
        png_img = mpimg.imread(input_image)
        png_plot = plt.imshow(png_img)
        plt.show()
    else:
        print("Error: File format not recognized. Must be .IMG or .png.")

def convert2_png(input_directory, output_directory):
    """ saves the input .img file as a .png, if a directory is given, iterates through all .img files in directory and saves as .png in the optional directory given. """
    file_count = 0
    # Is it a directory or single file?
    if os.path.isdir(input_directory):
        images = get_imgs(input_directory)

        for file in images:
            file_count += 1
            img_file = os.path.join(input_directory, file)
            image_array, bands = read_img(img_file)
            normalized_image = normalize(image_array)
            if bands > 1:
                plt.imsave(os.path.join(output_directory, Path(file).stem + '.png'), normalized_image, format='png')
            else:
                plt.imsave(os.path.join(output_directory, Path(file).stem + '.png'), image_array, format='png', cmap='gray')

    else:
        if input_directory.endswith(".IMG"):
            image_array, bands = read_img(input_directory)
            normalized_image = normalize(image_array)
            if bands > 1:
                plt.imsave(os.path.join(output_directory, Path(input_directory).stem + '.png'), normalized_image, format='png')
            else:
                plt.imsave(os.path.join(output_directory, Path(input_directory).stem + '.png'), image_array, format='png', cmap='gray')

    if file_count:
        print(file_count, "Files were converted")
    else:
        print("Image converted successfully")

def get_filetags(image_name):
    """ Given an image, returns relevant information from the filename such as which stereo pair, which engineering camera and what level of processing has been done. """

    stereo_tag = ""
    camera_tag = ""
    image_processing_code = ""
    try:
        image_name = image_name.split('_')
        if image_name[0][0].upper() == "N":
            camera_tag = "Navigation Camera"
        elif image_name[0][0].upper() == "Z":
            camera_tag = "Mast Camera"
        elif image_name[0][0].upper() == "F":
            camera_tag = "Front Camera"
        elif image_name[0][0].upper() == "R":
            camera_tag = "Rear Camera"
        else:
            camera_tag = "N/A"

        if image_name[0][1].upper() == "L":
            stereo_tag = "Left"
        elif image_name[0][1].upper() == "R":
            stereo_tag = "Right"
        else:
            stereo_tag = "N/A"

        image_processing_code = image_name[3][3:6]


        return camera_tag, stereo_tag, image_processing_code
    except IndexError:
        print("Input path not of expected type. Please double check input.")

def sort_stereo(input_directory):
    """ Given a directory, sorts all stereo images into new directories for left and right cameras. Creates a copy, does not move original files. """

    images = get_imgs(input_directory)
    left_dir_name = "Camera_Left"
    right_dir_name = "Camera_Right"
    print("Making new directories for Left and Right stereo pairs...")
    left_path = os.path.join(input_directory, left_dir_name)
    right_path = os.path.join(input_directory, right_dir_name)
    os.mkdir(left_path)
    os.mkdir(right_path)

    for im in images:
        print(im)
        camera, stereo_pair, image_code = get_filetags(im)
        #print(camera, stereo_pair, image_code)

        existing_imgpath = os.path.join(input_directory, im)
        existing_xmlpath = os.path.join(input_directory, im + ".xml")

        try:
            if stereo_pair.lower() == "left":

                shutil.copy(existing_imgpath, left_path)
                shutil.copy(existing_xmlpath, left_path)

            elif stereo_pair.lower() == "right":

                shutil.copy(existing_imgpath, right_path)
                shutil.copy(existing_xmlpath, right_path)
            else:
                print(im, "File was not moved as it's not of expected type.")
        except FileNotFoundError:
            print("Could not move", existing_xmlpath, im, "File may not exist.")


def find_stereo(image_path):
    """ Given a left or right camera image, returns the filename of the other pair. If no matching stereo pair is found, returns None. """
    #To Do: Depending on tests, must dynamically search for stereo pair in directory and not just edit filename for other pair.

    image_name = image_path.split("/")[-1]
    print("Given File:", image_name)
    camera, stereo_pair, image_code = get_filetags(image_name)

    if stereo_pair.lower() == "left":
        im = image_name.split('_')

        right_pair = im[0].replace("L", "R")

        im[0] = right_pair
        new_file = "_".join(im)
        print("Stereo Pair:", new_file)
        return new_file

    elif stereo_pair.lower() == "right":
        im = image_name.split('_')
        left_pair = im[0].replace("R", "L")

        im[0] = left_pair
        new_file = "_".join(im)
        print("Stereo Pair:", new_file)
        return new_file
    else:
        print("File name not of expected type.")
        return None

