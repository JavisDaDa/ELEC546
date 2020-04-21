
%Grad Credits: Photometric Stereo for Lambertian object
close all;
clc;

load('PhotometricStereo\Code\lighting.mat');
[imNum, imDim] = size(L);
imCategory = 'buddha';
imDirectory = ['PhotometricStereo\psmImages\', imCategory, '\'];

[imSet, imGray, mask] = LoadImages(imCategory, imDirectory, imNum);
[N, A, D] = PhotometricStereo(imSet, imGray, mask, L);

figure(2);
ARed = A(:, :, 1);
AGreen = A(:, :, 2);
ABlue = A(:, :, 3);
subplot 131, imshow(ARed);
subplot 132, imshow(AGreen);
subplot 133, imshow(ABlue);

figure(3);
[m, n, d] = size(N);
p = N(:, :, 1) ./ N(:, :, 3);
q = N(:, :, 2) ./ N(:, :, 3);
quiver(1:5:n, 1:5:m, p(1:5:end, 1:5:end), q(1:5:end, 1:5:end), 5);
axis ij;

figure(4); 
surfl(-D);
shading interp; colormap gray; axis tight;

figure(5)
depth = refineDepthMap(N, mask);
surfl(depth); shading interp; colormap gray; axis tight;



