%%  SFM Trial
%% Read Image
left = imread('ZL0_0050_0671382043_081ECM_N0031950ZCAM08013_063085J01.png');
right = imread('ZR0_0050_0671382043_081ECM_N0031950ZCAM08013_063085J01.png');
%% %% Convert Image from RGB 2 Grayscale for the matching algorithm
ImL = im2double(left);
ImL = rgb2gray(ImL);

ImR = im2double(right);
ImR = rgb2gray(ImR);
 %% Load Computed Parameters
load('CameraParamsMastCam.mat');
load('stereoParamsM.mat');
%% detect MinEigen Features Algorithm
doEigen = true;
if doEigen
    roi = [30, 30, size(ImL, 2) - 30, size(ImL, 1) - 30];
    imagePoints1 = detectMinEigenFeatures(im2gray(ImL), 'ROI', roi, ...
        'MinQuality', 0.001);
%% Creating a tracker point in image SFM
    tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 5);
    imagePoints1 = imagePoints1.Location;
    initialize(tracker, imagePoints1, ImL);

    [imagePoints2, validIdx] = step(tracker, ImR);
    matchedPoints1 = imagePoints1(validIdx, :);
    matchedPoints2 = imagePoints2(validIdx, :);
    showMatchedFeatures(ImL, ImR, matchedPoints1, matchedPoints2);
else
  pts1 = detectSURFFeatures(ImL,'MetricThreshold',1);
  pts2 = detectSURFFeatures(ImR,'MetricThreshold',1);
  [featuresL, valid_pt1] = extractFeatures(ImL, pts1);
    [featuresR, valid_pt2] = extractFeatures(ImR, pts2);
    
    indexPairs = matchFeatures(featuresL, featuresR);

    matchedPoints1 = valid_pt1(indexPairs(:,1),:);
    matchedPoints2 = valid_pt2(indexPairs(:,2),:);
    showMatchedFeatures(ImL, ImR, matchedPoints1, matchedPoints2);
end
%% Estimating Essential Matrix Utilising LeftMastCam as a reference
[E,status] = estimateEssentialMatrix(matchedPoints1, matchedPoints2,CameraParamsLM);
[orient, trans] = relativeCameraPose(E, CameraParamsLM, matchedPoints1, matchedPoints2);

disp(orient);         %the same as the values from the CAHVOR model
disp(trans);              %only up to scale
%% Using Left Camera pose as the reference 
tform1 = rigid3d;
camMatrix1 = cameraMatrix(CameraParamsLM, tform1);

cameraPose = rigid3d(orient, trans);
tform2 = cameraPoseToExtrinsics(cameraPose);
camMatrix2 = cameraMatrix(CameraParamsRM, tform2);

points3D = triangulate(matchedPoints1, matchedPoints2, camMatrix1, camMatrix2);
% numPixels = size(ImL, 1) * size(ImL, 2);
% allColors = reshape(ImL,[numPixels, 3]);
% if doEigen
%     colorIdx = sub2ind([size(ImL, 1), size(ImL, 2)], round(matchedPoints1(:,2)), ...
%     round(matchedPoints1(:, 1)));
% else
%     colorIdx = sub2ind([size(ImL, 1), size(ImL, 2)], round(matchedPoints1.Location(:,2)), ...
%         round(matchedPoints1.Location(:, 1)));
% end
% color = allColors(colorIdx, :);
% 
% % Create the point cloud
ptCloud = pointCloud(points3D);
% figure
pcshow(ptCloud)
%%
cameraSize = 0.3;
figure
plotCamera('Size', cameraSize, 'Color', 'r', 'Label', '1', 'Opacity', 0);
hold on
grid on
plotCamera('Location', trans, 'Orientation', orient, 'Size', cameraSize, ...
    'Color', 'b', 'Label', '2', 'Opacity', 0);

% Visualize the point cloud
pcshow(ptCloud, 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 1);

% Rotate and zoom the plot
camorbit(0, -30);
camzoom(1.5);

% Label the axes
xlabel('x-axis');
ylabel('y-axis');
zlabel('z-axis')

title('Up to Scale Reconstruction of the Scene');

