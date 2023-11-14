%% Calibrated 3D Reconstruction Method 1 MASTCAM
% BY : Nadine Elkhouly
% Email : nadineelkhouly@hotmail.com / naden.elkhouly@tu-dortmund.de
% ROVER: MARS 2020
%12/02/2022 _ UPDATED 10_03_2022
% ENSURE GetrotMatrix.m and ParamsMastCam.m is in the same directory
%% Loading Required Parameters
% load('CAHVORE.mat')
%% MARS 2020 MAST CAMERAS
C_ML = [0.745744,.448755,-1.9832];
A_ML = [0.81111,-0.556366,0.180481];
H_ML = [5602.5,6580.7,79.2549];
V_ML = [-812.42,663.422,8546.01];
R_ML = [-0.000325143,-0.35047,-1.90123];

% Right MAST CAMERA : {Vector axis,Vector Horizontal, Vector Vertical}
C_MR =[0.890717,0.645472,-1.98222];
A_MR =[0.788099,-0.588894,0.17922];
H_MR =[5843.62,6357.52,238.564];
V_MR = [-861.037,549.148,8545.14];
R_MR = [-5.83333e-05,-0.346709,-1.11689];
%Obtain Camera Parameters Object
ParamsMastCam2(A_ML',H_ML',V_ML',A_MR',H_MR',V_MR',C_ML',C_MR',R_ML',R_MR') 
 %% Load Computed Parameters
load('CameraParamsMastCam.mat');
load('stereoParamsM.mat');
%%  Read Image
left = imread('ZL0_0050_0671382043_081ECM_N0031950ZCAM08013_063085J01.png');
right = imread('ZR0_0050_0671382043_081ECM_N0031950ZCAM08013_063085J01.png');
%% Rectify Images : Calibrated Rectification
[J1,J2] = rectifyStereoImages(left,right,stereoParamsM);
figure 
imshowpair(J1,J2)
%% Compute rectified disparity
disparityMap1 = disparityBM(rgb2gray(J1),rgb2gray(J2),'Blocksize',5);
figure
imshow(disparityMap1,[]);
title('Disparity Map with block size of 5 CALIBRATED')
%% Disparity Coloured
disparityMap2 = disparitySGM(rgb2gray(J1),rgb2gray(J2));
figure; 
imshow(disparityMap2,[]);
colormap jet;   
colorbar;
%% Point Cloud Reconstruction 3D PCLD
points3D = reconstructScene(disparityMap2, stereoParamsM);
points3D = points3D ./ 1000;
ptCloud = pointCloud(points3D, 'Color', J1);


%% reconstruct the scene: 3D PCLD
xyzPoints = reconstructScene(disparityMap2,stereoParamsM);
player3D = pcplayer([-1 1], [-1, 1], [-1, 1], 'VerticalAxis', 'y', ...
    'VerticalAxisDir', 'down');
title(' CALIBRATED RECONSTRUCTION')
view(player3D, ptCloud);
% Coloured PCLD
figure
pcshow(xyzPoints)
title(' CALIBRATED RECONSTRUCTION')
xlabel('x')
ylabel('y')
zlabel('z')
ptCloud = pointCloud(xyzPoints,'Color',J1);
 %% Extract Point Cloud to a .ply file 
pcwrite(ptCloud,'MASTCAM','PLYFormat','binary');
% player3D = pcplayer([-3,3],[-3,3],[0,10],'VerticalAxis','x','VerticalAxisDir','down');
% view(player3D,ptCloud);

