import numpy as np
import math
import cv2
from get_cahvor import compute_rotation_matrix
import xml.etree.ElementTree as et


class PhotogrammetricModel(object):

    def __init__(self):
        tree = et.parse('NLF_0015_0668287694_287EBY_N0030386NCAM02001_13_0LLJ03.xml')
        root = tree.getroot()
        print(root[1][7][1][1][10][2][0][0].text)

        cahvore = dict([])
        mat = []
        name = 'CAHOVRE'
        for i in range(0,7):
            for j in range(0,3):
                mat.append(root[1][7][1][1][10][2][i][j].text)
            cahvore[name[i]] = mat
            mat = []

        self.C = cahvore['C']
        self.C = [float(x) for x in self.C]
        self.A = cahvore['A']
        self.A = [float(x) for x in self.A]
        self.H = cahvore['H']
        self.H = [float(x) for x in self.H]
        self.O = cahvore['O']
        self.O = [float(x) for x in self.O]
        self.V = cahvore['V']
        self.V = [float(x) for x in self.V]
        self.R = cahvore['R']
        self.R = [float(x) for x in self.R]

        self.pixelsize = float(input('Enter pixel size in mm: '))
        self.imsize = input('Enter Image Size (Row, Column): ')
        self.imsize = [float(x) for x in self.imsize.split()]

        a = np.array([self.A])
        h = np.array([self.H])
        v = np.array([self.V])
        
        self.hs = np.linalg.norm(np.cross(a,h))
        self.vs = np.linalg.norm(np.cross(a,v))
        self.hc = np.dot(a,h.T)
        self.vc = np.dot(a,v.T)
        print("")

        cahvor_model = dict([('C', self.C), ('A', self.A),
                             ('H', self.H), ('V', self.V),
                             ('O', self.O), ('R', self.R),
                             ('hs', self.hs), ('vs', self.vs),
                             ('hc', self.hc), ('vc', self.vc),
                             ('imsize', self.imsize),
                             ('pixel_size', self.pixelsize),
                             ])
        self.photogrammetric = compute_photogrammetric(cahvor_model)
        print_photogrammetric(self.photogrammetric)


def make_rotation_matrix(A, H, V, hs, vs, hc, vc):
    """
    ############################################
    # self.rotation = Rotational Matrix M      #
    #                                          #
    #                 [  H' ]                  #
    #             M = [ -V' ]                  #
    #                 [ -A  ]                  #
    #                                          #
    ############################################
    """

    # Compute H'
    H_n = (np.array(H) - (hc * np.array(A))) / hs

    # Compute V'
    V_n = (np.array(V) - (vc * np.array(A))) / vs

    # Rotational Matrix M generation
    r_matrix = np.zeros((3, 3))
    r_matrix[0, :] = H_n
    r_matrix[1, :] = - V_n
    r_matrix[2, :] = - np.array(A)

    return r_matrix


def compute_photogrammetric(CAHVOR_model):
    """
    Computation of photogrammetric parameters form CAHVOR.

    Parameters
    ----------
    CAHVOR: dict
        Take dictionary containing CAHVOR model and other parameters such as
        'hs', 'vs', 'hc' and 'vc'.

    Returns:
    photogrammetric: dict
        Returns dict containing computed photogrammetric parameters from
        CAHVOR model. Photogrammetric camera Parameters such as
        'camera center', 'focallength', 'rotation angles', 'rotation matrix',
        'pixel size', 'principal point', 'image size' and 'az' and 'el'
        to get back to origin position of PTU.
    """
    r_matrix = make_rotation_matrix(CAHVOR_model['A'], CAHVOR_model['H'],
                                    CAHVOR_model['V'], CAHVOR_model['hs'],
                                    CAHVOR_model['vs'], CAHVOR_model['hc'],
                                    CAHVOR_model['vc'])
    M = r_matrix
    f = CAHVOR_model['pixel_size'] * CAHVOR_model['hs']

    # camera center
    Xc = CAHVOR_model['C'][0]
    Yc = CAHVOR_model['C'][1]
    Zc = CAHVOR_model['C'][2]

    # angles
    phi = math.asin(r_matrix[2][0])
    w = - math.asin(r_matrix[2][1] / math.cos(phi))
    k = math.acos(r_matrix[0][0] / math.cos(phi))

    w = math.degrees(w)
    phi = math.degrees(phi)
    k = math.degrees(k)

    k0 = CAHVOR_model['R'][0]
    k1 = CAHVOR_model['R'][1] / (f**2)
    k2 = CAHVOR_model['R'][2] / (f**4)

    x0 = CAHVOR_model['pixel_size'] * \
        (CAHVOR_model['hc'] - (CAHVOR_model['imsize'][1] / 2))
    y0 = - CAHVOR_model['pixel_size'] * \
        (CAHVOR_model['vc'] - (CAHVOR_model['imsize'][0] / 2))
    R = compute_rotation_matrix(w, phi, k)

    photogrammetric = dict([('M', M), ('f', f), ('Xc', Xc), ('Yc', Yc),
                            ('Zc', Zc), ('w', w), ('phi', phi), ('k', k),
                            ('k0', k0), ('k1', k1), ('k2', k2), ('x0', x0),
                            ('y0', y0), ('R', R)])
    photogrammetric_para = cv2.FileStorage('photo_para.xml', cv2.FILE_STORAGE_WRITE)
    photogrammetric_para.write('Focal', photogrammetric['f'])
    photogrammetric_para.write('xc', photogrammetric['Xc'])
    photogrammetric_para.write('yc', photogrammetric['Yc'])
    photogrammetric_para.write('zc', photogrammetric['Zc'])
    photogrammetric_para.write('xo', photogrammetric['x0'])
    photogrammetric_para.write('yo', photogrammetric['y0'])
    photogrammetric_para.write('w', photogrammetric['w'])
    photogrammetric_para.write('p', photogrammetric['phi'])
    photogrammetric_para.write('k', photogrammetric['k'])
    photogrammetric_para.write('k0', photogrammetric['k0'])
    photogrammetric_para.write('k1', photogrammetric['k1'])
    photogrammetric_para.write('k2', photogrammetric['k2'])

    return photogrammetric


def print_photogrammetric(photogrammetric):
    print("--------------------------------------------------------------")
    print("")
    print("f: ", photogrammetric['f'])
    print("Xc: ", photogrammetric['Xc'])
    print("Yc: ", photogrammetric['Yc'])
    print("Zc: ", photogrammetric['Zc'])
    print("x0: ", photogrammetric['x0'])
    print("y0: ", photogrammetric['y0'])

    # print('Rotation Matrix: ', photogrammetric['M'])
    print('Angle w (deg): ', photogrammetric['w'])
    print('Angle phi (deg): ', photogrammetric['phi'])
    print('Angle k (deg): ', photogrammetric['k'])

    print('k0: ', photogrammetric['k0'])
    print('k1: ', photogrammetric['k1'])
    print('k2: ', photogrammetric['k2'])
    print("")
    print("--------------------------------------------------------------")

if __name__ == '__main__':
    camera_matrix = PhotogrammetricModel()
