close all; clear all; clc
%%
load('cam1_1.mat');
load('cam2_1.mat');
load('cam3_1.mat');

%%
% get the dimension of loaded file
% all and b11 = size of image
% c11 means layers of rgm image
% d11 = number of images in the vids
[a1, b1 , c1 , d1 ] = size(vidFrames1_1) ;

% get the basic idea of where the osscilation is 
% use for creating small window 
% imshow(rgb2gray(vidFrames1_1(:,:,:,1)))

XY1 =[];
for i = 1:d1
   
    %change each color image into gray-scale
   img = rgb2gray(vidFrames1_1(:,:,:,i));
   
   % blacked out the background environmen
   % leave the small window
   img(:,1:250) = 0;
   img(:,450:end) = 0;
   img(1:202,:) = 0;
   img(440:end,:) = 0;

   %imshow(img)
   
   % locate the coordinates of the mass
   thresh = img(:) > 250;
   indeces = find(thresh);
   [y, x] = ind2sub(size(img),indeces);
   XY1 = [XY1; mean(x), mean(y)];

end


%% for cam_2
[a2, b2 , c2 , d2 ] = size(vidFrames2_1);
% get the basic idea of where the osscilation is 
% use for creating small window 
% imshow(rgb2gray(vidFrames2_1(:,:,:,1)))

XY2 = [];
for i = 1:d2
   
    %change each color image into gray-scale
   img = rgb2gray(vidFrames2_1(:,:,:,i));
   
   % blacked out the background environmen
   % leave the small window
   img(:,1:240) = 0;
   img(:,350:end) = 0;
   img(370:end,:) = 0;
   %imshow(img)
   
   % locate the coordinates of the mass
   thresh = img(:) > 250;
   indeces = find(thresh);
   [y, x] = ind2sub(size(img),indeces);
   XY2 = [XY2; mean(x), mean(y)];

end

%% for cam_3
[a3, b3 , c3 , d3 ] = size(vidFrames3_1);

% get the basic idea of where the osscilation is 
% use for creating small window 
% imshow(rgb2gray(vidFrames3_1(:,:,:,1)))


XY3 = [];
for i = 1:d3
   
   %change each color image into gray-scale
   img = rgb2gray(vidFrames3_1(:,:,:,i));
   
   % blacked out the background environmen
   % leave the small window
   img(:,1:250) = 0;
   img(:,500:end) = 0;
   img(1:230,:) = 0;
   img(336:end,:) = 0;
   %imshow(img)
   
   % locate the coordinates of the mass
   thresh = img(:) > 248;
   indeces = find(thresh);
   [y, x] = ind2sub(size(img),indeces);
   XY3 = [XY3; mean(x), mean(y)];

end

%% Plot the result
figure(1)
subplot(3,2,1)
plot(XY1(:,1))
ylabel('Position in X')
ylim([0, 500])
xlim([0, 250])
title('Cam1')

subplot(3,2,2)
plot(XY1(:,2))
ylabel('Position in Y')
ylim([0, 500])
xlim([0, 250])
title('Cam1')

subplot(3,2,3)
plot(XY2(:,1))
ylabel('Position in X')
ylim([0, 500])
xlim([0, 250])
title('Cam2')

subplot(3,2,4)
plot(XY2(:,2))
ylabel('Position in Y')
ylim([0, 500])
xlim([0, 250])
title('Cam2')

subplot(3,2,5)
plot(XY3(:,1))
ylabel('Position in X')
ylim([0, 500])
xlim([0, 250])
title('Cam3')

subplot(3,2,6)
plot(XY3(:,2))
ylabel('Position in Y')
ylim([0, 500])
xlim([0, 250])
title('Cam3')

%%
% to do SDV, we have to make a matrix in the same size
% so we cut the other two matrixs to the smallest one
min_len = min([length(XY1(:,1)), length(XY2(:,1)), length(XY3(:,1))]);

XY1 = XY1(1:min_len,:);
XY2 = XY2(1:min_len,:);
XY3 = XY3(1:min_len,:);

alldata = [XY1';XY2';XY3'];

%%
[m,n]=size(alldata); % compute data size
mn=mean(alldata,2); % compute mean for each row
alldata=alldata-repmat(mn,1,n); % subtract mean

[u,s,v]=svd(alldata'/sqrt(n-1)); % perform the SVD
lambda=diag(s).^2; % produce diagonal variances

Y= alldata' * v; % produce the principal components projection
sig=diag(s);

%%
figure()
plot(1:6, lambda/sum(lambda), 'bo--', 'Linewidth', 2);
title("Ideal Case: Energy of each Diagonal Variance");
xlabel("Diagonal Variances"); ylabel("Energy Captured");

figure()
plot(1:min_len, Y(:,1), 'Linewidth', 2)
ylabel("Displacement (pixels)"); xlabel("Time (frames)"); 
title("Ideal Case: Displacement across principal component directions");
legend("PC1")
