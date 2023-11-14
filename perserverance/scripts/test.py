import calibration as calib
import cv2 as cv

imagel = cv.imread("lefrncam_57.png", cv.IMREAD_GRAYSCALE)
imager = cv.imread("rightrncam_57.png", cv.IMREAD_GRAYSCALE)

key1, key2 = calib.draw_keypoints_and_match(imagel, imager)
f = calib.calculate_F_matrix(key1, key2)
print(f)
