function []= IntrinsicsNavCam(A_NL,H_NL,V_NL,A_NR,H_NR,V_NR,C_NL,C_NR,R_NL,R_NR)
%% Function Converting CAHVORE Camera to Pinhole model FOR NavCams FOR M2020 ROVER
% By : Naden Elkhouly 
% Email : nadineelkhouly@hotmail.com
% 15/02/2022
%% Input Parameters
% pixel size in mm for NavCam
Pixel_size = 0.0255
% Image size
img_size = [976,1296];
image_width = 976;
image_height = 1296;

%% RIGHT NAVCAM PARAMETERS COMPUTATION
% section computing hs hc Vs and Vc for RIGHT NAVCAM
hsR = norm(cross(A_NR,H_NR))
vsR =  norm(cross(A_NR,V_NR))
hcR = dot(A_NR,H_NR)
vcR = dot(A_NR,V_NR)
% Compute H' and V'
H_nR = (H_NR- (hcR * A_NR)) / hsR  
V_nR = (V_NR- (vcR * A_NR)) / vsR 
%% Rotation Matrix FOR RIGHT CAMERA
r_matrix_right = zeros(3,3);
r_matrix_right(1,:) = H_nR
r_matrix_right(2,:) = -V_nR
r_matrix_right(3,:) = -A_NR
M_right = r_matrix_right
%% Rotation angles in degrees for RIGHT NAVCAM
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
    XcR = C_NR(1)
    YcR= C_NR(2)
    ZcR = C_NR(3)
 %% Intrinsics for Right NAV Camera
f_R = Pixel_size *hsR
Cx_NR = (hcR-(image_width/2))*Pixel_size
Cy_NR = -(vcR-(image_height/2))*Pixel_size
% setting skew parameter equal to 0 as image axis are perpendicular  
S = 0;
%% Radial Lens Distortion coeff for RIGHT NAVCAM
k0R = R_NR(1)
k1R = R_NR(2)/ (f_R^2)
k2R = R_NR(3)/ (f_R^4)

%% LEFT NAVCAM PARAMETERS COMPUTATION
%% Computing hs hc vs and vc for LEFT NAVCAM
hsL= norm(cross(A_NL,H_NL))
vsL =  norm(cross(A_NL,V_NL))
hcL = dot(A_NL,H_NL)
vcL = dot(A_NL,V_NL)
% H' for LEFT NAVCAM
H_nL = (H_NL- (hcL * A_NL)) / hsL
V_nL = (V_NL- (vcL * A_NL)) / vsL
%% Rotation Matrix for LEFT NAVCAM
r_matrix_left = zeros(3,3);
r_matrix_left(1,:) = H_nL
r_matrix_left(2,:) = -V_nL
r_matrix_left(3,:) = -A_NL
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
XcL = C_NL(1)
YcL= C_NL(2)
ZcL = C_NL(3)
   
%% Intrinsics for Left NAV Camera
% Focal length left in pixels
f_L = Pixel_size *hsL
% Principle Point calculation 
Cx_NL = (hcL-(image_width/2))*Pixel_size
Cy_NL = -(vcL-(image_height/2))*Pixel_size

%% Radial Lens Distortion coeff for LEFT NAVCAM
k0L = R_NL(1)
k1L = R_NL(2)/ (f_L^2)
k2L = R_NL(3)/ (f_L^4)

%% Left Camera Parameters Object
% 3x3 Intrinsic Matrix
IntrinsicMatrixLeft = [f_L,0,0; S,f_L,0; Cx_NL, Cy_NL,1];
radialDistortionL = [k0L,k1L,k2L]; 
CameraParamsLN= cameraParameters('IntrinsicMatrix',IntrinsicMatrixLeft,'ImageSize',img_size,'RadialDistortion',radialDistortionL); 
%CameraParamsL.RotationMatrices = tform_Navcam_Left.Rotation ;
%% Creating Right Camera Parameters Object
% CR1= cameraParameters
% 3x3 Intrinsic Matrix
IntrinsicMatrixRight = [f_R,0,0; S,f_R,0; Cx_NR, Cy_NR,1];
radialDistortionR = [k0R,k1R,k2R]; 
CameraParamsRN= cameraParameters('IntrinsicMatrix',IntrinsicMatrixRight,'ImageSize',img_size,'RadialDistortion',radialDistortionR); 
save('CameraParamsNavCam.mat','CameraParamsLN','CameraParamsRN');
end