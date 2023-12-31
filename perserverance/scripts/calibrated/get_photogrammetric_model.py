import numpy as np
import math
from get_cahvor import compute_rotation_matrix


class PhotogrammetricModel(object):

    def __init__(self):
        print("--------------------------------------------------------------")
        print('Note: Enter Vector Elements Row by Row with Space in between')
        print("--------------------------------------------------------------")
        print("")
        self.C = input('Enter C Vector: ')
        self.C = [float(x) for x in self.C.split()]
        self.A = input('Enter A Vector: ')
        self.A = [float(x) for x in self.A.split()]
        self.H = input('Enter H Vector: ')
        self.H = [float(x) for x in self.H.split()]
        self.V = input('Enter V Vector: ')
        self.V = [float(x) for x in self.V.split()]
        self.O = input('Enter O Vector: ')
        self.O = [float(x) for x in self.O.split()]
        self.R = input('Enter R Vector: ')
        self.R = [float(x) for x in self.R.split()]
        self.pixelsize = float(input('Enter pixel size in mm: '))
        self.imsize = input('Enter Image Size (Row, Column): ')
        self.imsize = [float(x) for x in self.imsize.split()]

        self.hs = float(input('Enter hs: '))
        self.vs = float(input('Enter vs: '))
        self.hc = float(input('Enter hc: '))
        self.vc = float(input('Enter vc: '))
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
    f = CAHVOR_model['hs']

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

    x0 = round(CAHVOR_model['hc'])
    y0 = - round(CAHVOR_model['vc'])
    R = compute_rotation_matrix(w, phi, k)
    print(R)

    photogrammetric = dict([('M', M), ('f', f), ('Xc', Xc), ('Yc', Yc),
                            ('Zc', Zc), ('w', w), ('phi', phi), ('k', k),
                            ('k0', k0), ('k1', k1), ('k2', k2), ('x0', x0),
                            ('y0', y0), ('R', R)])
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
