img = double(imread('camera_man_noisy.png'))/255;
img(img < 0) = 0;
img(img > 1) = 1;
w = 15;
sig = [7758 7758];
bflt = bfilter2(img, w, sig);
% Display grayscale input image and filtered output.
figure(1); clf;
set(gcf,'Name','Grayscale Bilateral Filtering Results');
subplot(1,2,1); imagesc(img);
axis image; colormap gray;
title('Input Image');
subplot(1,2,2); imagesc(bflt);
axis image; colormap gray;
title('Result of Bilateral Filtering');