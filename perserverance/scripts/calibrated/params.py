#!/bin/python3
import get_photogrammetric_model as photoModel
import utilsM2020 as utils
import numpy as np
import os

class Params:
    def __init__(self, xmls, pixelSize, imageSize):
        for xml in xmls:
            data = utils.get_cahvore(xml)
            if xml[0:2] == 'ZL':
                self.cL = np.array(data['Vector_Center'], dtype='float64')
                self.aL = np.array(data['Vector_Axis'], dtype='float64')
                self.hL = np.array(data['Vector_Horizontal'], dtype='float64')
                self.vL =  np.array(data['Vector_Vertical'], dtype='float64')
                self.oL = np.array(data['Vector_Optical'], dtype='float64')
                self.rL = np.array(data['Radial_Terms'], dtype='float64')
                self.hsL = np.linalg.norm(np.cross(self.aL, self.hL))
                self.vsL = np.linalg.norm(np.cross(self.aL, self.vL))
                self.hcL = np.dot(self.aL, self.hL)
                self.vcL = np.dot(self.aL, self.vL)
                self.imSize = imageSize
                self.pxSize = pixelSize
                inputParametersL = dict([('C', self.cL), ('A', self.aL), ('H', self.hL), ('V', self.vL),
                    ('O', self.oL), ('R', self.rL), ('hs', self.hsL), ('vs', self.vsL), ('hc', self.hcL),
                    ('vc', self.vcL), ('imsize', self.imSize), ('pixel_size', self.pxSize)])
                self.intrinsicParametersLeft = photoModel.compute_photogrammetric(inputParametersL)
                self.cameraMatrixL = np.array([
                    [self.intrinsicParametersLeft['f'], 0, self.intrinsicParametersLeft['x0']],
                    [0, self.intrinsicParametersLeft['f'], self.intrinsicParametersLeft['y0']],
                    [0, 0, 1]], dtype='float64')
                self.distL = np.array([self.intrinsicParametersLeft['k0'], self.intrinsicParametersLeft['k1'],
                    0, 0, self.intrinsicParametersLeft['k2']], dtype='float64')
                
            else:
                self.cR = np.array(data['Vector_Center'], dtype='float64')
                self.aR = np.array(data['Vector_Axis'], dtype='float64')
                self.hR = np.array(data['Vector_Horizontal'], dtype='float64')
                self.vR =  np.array(data['Vector_Vertical'], dtype='float64')
                self.oR = np.array(data['Vector_Optical'], dtype='float64')
                self.rR = np.array(data['Radial_Terms'], dtype='float64')
                self.hsR = np.linalg.norm(np.cross(self.aR, self.hR))
                self.vsR = np.linalg.norm(np.cross(self.aR, self.vR))
                self.hcR = np.dot(self.aR, self.hR)
                self.vcR = np.dot(self.aR, self.vR)
                self.imSize = imageSize
                self.pxSize = pixelSize
                inputParametersR = dict([('C', self.cR), ('A', self.aR), ('H', self.hR), ('V', self.vR),
                    ('O', self.oR), ('R', self.rR), ('hs', self.hsR), ('vs', self.vsR), ('hc', self.hcR),
                    ('vc', self.vcR), ('imsize', self.imSize), ('pixel_size', self.pxSize)])
                self.intrinsicParametersRight = photoModel.compute_photogrammetric(inputParametersR)
                self.cameraMatrixR = np.array([
                    [self.intrinsicParametersRight['f'], 0, self.intrinsicParametersRight['x0']],
                    [0, self.intrinsicParametersRight['f'], self.intrinsicParametersRight['y0']],
                    [0, 0, 1]], dtype='float64')
                self.distR = np.array([self.intrinsicParametersRight['k0'], self.intrinsicParametersRight['k1'],
                    0, 0, self.intrinsicParametersRight['k2']], dtype='float64')
        
        rLeft = self.intrinsicParametersLeft['R']
        rRight = self.intrinsicParametersRight['R']
        rotMat = rRight*np.linalg.inv(rLeft)
        transMat = self.cR - self.cL 
        extrinscis = dict([('R', rotMat), ('T', transMat)])
        self.extrinscis = extrinscis

