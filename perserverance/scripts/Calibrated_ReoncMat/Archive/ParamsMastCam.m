function []= ParamsMastCam(A_ML,H_ML,V_ML,A_MR,H_MR,V_MR,C_ML,C_MR,R_ML,R_MR)
%% Function Converting CAHVORE Camera to Photogrammetric Model FOR MastCams M2020 ROVER
% By : Nadine Elkhouly 
% 15/02/2022
% Email : nadineelkhouly@hotmail.com/ naden.elkhouly@tu-dortmund.de
%% Input Parameters
% pixel size in mm
Pixel_size = 0.074
% Image size
img_size = [1200,1648];
image_width =1200;
image_height = 1648;
%% RIGHT MASTCAM PARAMETERS COMPUTATION
% section computing hs hc Vs and Vc for RIGHT NAVCAM
hsR = norm(cross(A_MR,H_MR))
vsR =  norm(cross(A_MR,V_MR))
hcR = dot(A_MR,H_MR)
vcR = dot(A_MR,V_MR)
% Compute H' and V' H' for right MASTCAM
H_mR = (H_MR- (hcR * A_MR)) / hsR  
V_mR = (V_MR- (vcR * A_MR)) / vsR
%% Rotation Matrix FOR RIGHT CAMERA
r_matrix_right = zeros(3,3);
r_matrix_right(1,:) = H_mR
r_matrix_right(2,:) = -V_mR
r_matrix_right(3,:) = -A_MR
M_right = r_matrix_right
%% Rotation angles in degrees for RIGHT MASTCAM
% ROT Y
    phi_R = asin(M_right(3,1))
% ROT X
    w_R= -asin(M_right(3,2)/cos(phi_R))
 % ROT Z
    k_R =acos(M_right(1,1)/cos(phi_R))
% FROM radians2degrees
    w_R= rad2deg(w_R)
    phi_R= rad2deg(phi_R)
    k_R= rad2deg(k_R)
%% RIGHT Camera Center
    XcR = C_MR(1)
    YcR= C_MR(2)
    ZcR = C_MR(3)
%% Intrinsics for Right MAST Camera
% in m
f_R = (Pixel_size *hsR)
% Principle Point calculation 
Cx_MR = (hcR-(image_width/2))*Pixel_size
Cy_MR = -(vcR-(image_height/2))*Pixel_size
% setting skew parameter equal to 0 as image axis are perpendicular  
S = 0;
%% Radial Lens Distortion coeff for RIGHT MASTCAM
k0R = R_MR(1)
k1R = R_MR(2)/ (f_R^2)
k2R = R_MR(3)/ (f_R^4)
%% Rotation matrix for Right Mastcam
Rot_Right = GetrotMatrix(w_R,phi_R,k_R);
%% LEFT MASTCAM PARAMETERS COMPUTATION
%% Computing hs hc vs and vc for LEFT MASTCAM
hsL= norm(cross(A_ML,H_ML))
vsL =  norm(cross(A_ML,V_ML))
hcL = dot(A_ML,H_ML)
vcL = dot(A_ML,V_ML)
% H' for LEFT MASTCAM
H_mL = (H_ML- (hcL * A_ML)) / hsL
% V' for LEFT MASTCAM
V_mL = (V_ML- (vcL * A_ML)) / vsL
%% Rotation Matrix for LEFT MASTCAM
r_matrix_left = zeros(3,3);
r_matrix_left(1,:) = H_mL
r_matrix_left(2,:) = -V_mL
r_matrix_left(3,:) = -A_ML
M_left = r_matrix_left

%% angles in degrees LEFT : (w,phi,k) are the Eulerangle for rotations about the X, Y and Z axes in successionthat make up the world to camera rotation matrix M
% ROT Y
phiL = asin(M_left(3,1))
% ROT X
wL = -asin(M_left(3,2)/cos(phiL))
% ROT Z
kL =acos(M_left(1,1)/cos(phiL))
wL = rad2deg(wL)
phiL = rad2deg(phiL)
kL = rad2deg(kL)
%% LEFT Camera Center
XcL = C_ML(1)
YcL= C_ML(2)
ZcL = C_ML(3)   
%% Intrinsics for Left MAST Camera
% in m
f_L = (Pixel_size *hsL)
% Principle Point calculation 
Cx_ML = (hcL-(image_width/2))*Pixel_size
Cy_ML = -(vcL-(image_height/2))*Pixel_size
%% Radial Lens Distortion coeff for LEFT MASTCAM
k0L = R_ML(1)
k1L = R_ML(2)/ (f_L^2)
k2L = R_ML(3)/ (f_L^4)
%% Rotation matrix for Left Mastcam
Rot_Left = GetrotMatrix(wL,phiL,kL);

%% Extrinsics
rotation = Rot_Right*inv(Rot_Left);
tRight =C_MR ;
tLeft = C_ML;
translate = tRight-tLeft;
%% Left Camera Parameters Object
% 3x3 Intrinsic Matrix
IntrinsicMatrixLeft = [f_L,0,0; S,f_L,0; Cx_ML, Cy_ML,1];
radialDistortionL = [k0L,k1L,k2L];

CameraParamsLM = cameraParameters('IntrinsicMatrix',IntrinsicMatrixLeft,...
      'RotationVectors',Rot_Left,'TranslationVectors',diag(tLeft),'ImageSize',img_size,'RadialDistortion',radialDistortionL);
%% Creating Right Camera Parameters Object
% CR1= cameraParameters
% 3x3 Intrinsic Matrix
IntrinsicMatrixRight = [f_R,0,0; S,f_R,0; Cx_MR, Cy_MR,1];
radialDistortionR = [k0R,k1R,k2R]; 
CameraParamsRM= cameraParameters('IntrinsicMatrix',IntrinsicMatrixRight,'ImageSize',img_size,'RadialDistortion',radialDistortionR,'RotationVectors',Rot_Right,'TranslationVectors',diag(tRight)'); 
%% Stereo Params MASTCAM
% rot2 = [1,0,0;0,1,0;0,0,1]
% trans2 = [244,0,0]
% stereoParamsM =stereoParameters(CameraParamsRM,CameraParamsLM,rot2,trans2)
stereoParamsM = stereoParameters(CameraParamsLM,CameraParamsRM,rotation,translate);
%% SAVING Parameters
save('CameraParamsMastCam.mat','CameraParamsLM','CameraParamsRM');
save('stereoParamsM.mat','stereoParamsM');
end