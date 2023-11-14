# Author: Ashwin Nedungadi
# Date: 12.12.21
# Test Script for utils_m2020.py

import utils_m2020

#test_output_directory = "C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/Data/converted/"
#test_input_directory = "C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/Data/Test/"
#
#bwtest_img = "C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/Data/Test/NLB_0002_0667130127_539EDR_N0010052AUT_04096_00_2I3J02.IMG"
#bwtest_png = "C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/Data/MARS2020/SOL02/ncam/NLB_0002_0667130127_539EDR_N0010052AUT_04096_00_2I3J02.png"
#
#rgbtest_img = "C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/Data/Test/NLF_0002_0667130259_241EBY_N0010052AUT_04096_00_0LLJ02.IMG"
#rgbtest_png = "C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/Data/MARS2020/SOL02/ncam/NLF_0002_0667130259_241EBY_N0010052AUT_04096_00_0LLJ02.png"
#
#zcam_png = "C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/Data/MARS2020/SOL02/zcam/ZL0_0002_0667131112_000EDR_N0010052AUT_04096_0260LUJ02.png"
#
#bwtest_img_xml = "C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/Data/Test/NLB_0002_0667130127_539EDR_N0010052AUT_04096_00_2I3J02.xml"
#rgbtest_img_xml = "C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/Data/Test/NLF_0002_0667130259_241EBY_N0010052AUT_04096_00_0LLJ02.xml"
#
#
## Perform Tests for all cases
#
##VIEWING
#utils_m2020.view(bwtest_img)
#utils_m2020.view(bwtest_png)
#
#utils_m2020.view(rgbtest_img)
#utils_m2020.view(rgbtest_png)
#""" Q.C-script Does not crash or throw errors. """
#
##XML INFO
#utils_m2020.read_xml(bwtest_img_xml)
#print('-'*30)
#utils_m2020.read_xml(rgbtest_img_xml)
#""" Q.C-script Does not crash or throw errors. """
#
#
## CONVERTING TO .PNG
## Multiple .IMG FILES
#utils_m2020.convert2_png(test_input_directory, test_output_directory)
## Single .IMG FILE
#utils_m2020.convert2_png(bwtest_img, test_output_directory+'/single')
#utils_m2020.convert2_png(rgbtest_img, test_output_directory+'/single')
#""" Q.C-script Does not crash or throw errors. """
#
#
## Find Stereo Pair
#dir = "C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/Data/MARS2020/SOL02/ncam/"
## This is how error handling and calling should be done in main script
#for im in utils_m2020.get_imgs(dir):
#    pair = utils_m2020.find_stereo(im)
#    try:
#        utils_m2020.view(dir+pair)
#    except FileNotFoundError:
#        print("File", im, "wasn't found")
#""" Q.C-script Does not crash or throw errors. """
#
## Sorting all stereo pairs in given directory
#utils_m2020.sort_stereo("C:/Users/Ashwin/Desktop/WiSe 2021/Semester Project/Data/MARS2020/SOL02/rcam")
#""" Q.C-script Does not crash or throw errors. """

utils_m2020.sort_stereo('/home/jay/wise/project/perserverance/images')
