%% Extrinsic Parameters M2020: Navigation & Mast Cameras Poses Relative to RSM Head
% References and Software Version Requirements 
% Note: poseplot requires MATLAB R2021b 

% [1] https://naif.jpl.nasa.gov/pub/naif/MARS2020/misc/hlee/m2020_v03.tf
% [2] https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html#Defining%20a%20TK%20Frame%20Using%20Euler%20Angles
% [3] https://planet-vr.atlassian.net/wiki/spaces/PV/pages/24576061/Navigation+Cameras+Frames+wrt+RSM+Head

%% Navigation Cameras Pose Relative to RSM HEAD
% Rotation Matrices ( x = 1 , y= 2 , z= 3)
x = 0; y = pi/2; z = -pi/2;
[rotx,roty,rotz] = ang2rotm(x,y,z);

% Navcam angles order -> 2, 1, 3 (roty , rotx , rot z)

% Navcam Left camera Rotation  : Pre-Multiplication of rotation around ZXY
RotN_L = rotz*rotx*roty;
% Navcam Right camera Rotation :  Pre-Multiplication of rotation around ZXY
RotN_R = rotz*rotx*roty;

% Rotation matrix to homogenous Tform
tformN_L = rotm2tform(RotN_L);
tformN_R = rotm2tform(RotN_R);

% Add translation Left - adding half of Navcam baseline along Y axis of RSM head (Check Reference 3)
tform_Navcam_Left = rigid3d;
tform_Navcam_Left.Translation = [0, -21.2, 0];
tform_Navcam_Left.Rotation = RotN_L;

% Add translation Right - adding half of Navcam baseline along Y axis of RSM head (Check Reference 3)
tform_Navcam_Right = rigid3d;
tform_Navcam_Right.Translation = [0,21.2,0];
tform_Navcam_Right.Rotation = RotN_R

% Plot Navigation Camera Pose (LEFT & RIGHT)
xlabel("North-x (m)");
ylabel("East-y (m)");
zlabel("Down-z (m)");
p1 = poseplot(tform_Navcam_Left.Rotation, tform_Navcam_Left.Translation);
hold on
p2 = poseplot(tform_Navcam_Right.Rotation, tform_Navcam_Right.Translation);
legend("Navcam Left","Navcam Right")
hold off

%% Mast Cameras Pose Relative to RSM HEAD
% Rotation Matrices

% Left Mast Camera 
x_ml = deg2rad(-1.25); y_ml = pi/2; z_ml = -pi/2;
[rotx_ml,roty_ml,rotz_ml] = ang2rotm(x_ml, y_ml, z_ml);
% Right Mast Camera
x_mr = deg2rad(1.25); y_mr = pi/2; z_mr = -pi/2;
[rotx_mr, roty_mr, rotz_mr] = ang2rotm(x_mr, y_mr, z_mr);

% Mastcam angles order of Rotation -> 2, 1, 3 (roty, rotx, rotz)
% Mast camera Rotation  : Pre-Multiplication of rotation given angles(ZXY)
RotM_L = rotz_ml*rotx_ml*roty_ml;
RotM_R = rotz_mr*rotx_mr*roty_mr;

% Rotation matrix to homogenous Tform
tformM_L = rotm2tform(RotM_L);
tformM_R = rotm2tform(RotM_R);

% Transformation Matrix - Add translation Left - adding half of MastCam baseline along Y axis of RSM head (Check Reference 4)
% Rotation Matrix given in RADIANS 
% Translation vector -ve in Y
tform_Mastcam_Left = rigid3d
tform_Mastcam_Left.Translation = [0,-12.4,0];
tform_Mastcam_Left.Rotation = RotM_L ;

% % Transformation Matrix - Add translation Right - adding half of MastCam baseline along Y axis of RSM head (Check Reference 4)
% Rotation Matrix given in RADIANS
% Translation Vector +ve in Y
tform_Mastcam_Right = rigid3d
tform_Mastcam_Right.Translation = [0,12.4,0];
tform_Mastcam_Right.Rotation = RotM_R ;

% Plot Mast Camera Pose (LEFT & RIGHT)
xlabel("North-x (m)");
ylabel("East-y (m)");
zlabel("Down-z (m)");
p1 = poseplot(tform_Mastcam_Left.Rotation, tform_Mastcam_Left.Translation);
hold on
p2 = poseplot(tform_Mastcam_Right.Rotation, tform_Mastcam_Right.Translation);
legend("Mastcam Left","Mastcam Right")
hold off


% RSM Origin Frame
% Rotation Matrix in RADIANS
tform_Origin = rigid3d
tform_Origin.Translation = [0,0,0]
tform_Origin.Rotation = eye(3) % In radians

%% Plot all cameras and Origin Frame
xlabel("North-x (m)");
ylabel("East-y (m)");
zlabel("Down-z (m)");
p1 = poseplot(tform_Mastcam_Left.Rotation, tform_Mastcam_Left.Translation);
hold on
p2 = poseplot(tform_Mastcam_Right.Rotation, tform_Mastcam_Right.Translation);
p3 = poseplot(tform_Navcam_Left.Rotation, tform_Navcam_Left.Translation);
p4 = poseplot(tform_Navcam_Right.Rotation, tform_Navcam_Right.Translation);
p5 = poseplot(tform_Origin.Rotation, tform_Origin.Translation);
legend("Mastcam Left","Mastcam Right","Navcam Left","Navcam Right", "Origin")
hold off
 
%% Obtaining Stereo Pair : NAVIGATION CAMERAS
NavcamL_wrt_R = tform_Navcam_Left.T * tform_Navcam_Right.T';
NavcamR_wrt_L = tform_Navcam_Right.T * tform_Navcam_Left.T';

%% Obtaining Stereo Pair : MAST CAMERAS
MastcamL_wrt_R = tform_Mastcam_Left.T * tform_Mastcam_Right.T';
MastcamR_wrt_L = tform_Mastcam_Right.T * tform_Mastcam_Left.T';

%% Saving Extrinsic Parameters .mat Format (NavCameras , Origin , MastCameras)
save('NavCam_Tform.mat','tform_Navcam_Right','tform_Navcam_Left');
save('MastCam_Tform.mat','tform_Mastcam_Right','tform_Mastcam_Right');
save('RSM_Origin.mat','tform_Origin');

%%
tnl = tform_Navcam_Left.T
tnr = tform_Navcam_Right.T

trvec_nl = tform_Navcam_Left.Translation
trvec_nr = tform_Navcam_Right.Translation

rotm_nl = tform_Navcam_Left.Rotation
rotm_nr = tform_Navcam_Right.Rotation


