from params import *
import os

xmls = list()

for files in os.listdir():
    if files.endswith('.xml'):
        xmls.append(files)

pixel_size = 0.0074
img_size = [1200,1648];

parameters = Params(xmls, pixel_size, img_size)

print('--------------------------------------')
print(parameters.intrinsicParametersLeft)
print(parameters.distL)
print(parameters.intrinsicParametersRight)
print(parameters.distR)
print('--------------------------------------')
print(parameters.extrinscis)

