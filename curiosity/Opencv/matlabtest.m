imr = imread("right.png");
iml = imread("left.png");
disparityMap = disparitySGM(im2gray(iml),im2gray(imr));
imshow(disparityMap)