i = imread('barbara.jpg');
img = rgb2gray(i);
imshow(img);
f1 = [-1, 0, 1];
res1 = imfilter(img, f1);
imshow(res1);
imwrite(res1, 'Central.png');
f2 = fspecial('sobel');
res2 = imfilter(img, f2);
imshow(res2);
imwrite(res2, 'Sobel.png');
f3 = fspecial('average');
res3 = imfilter(img, f3);
imshow(res3);
imwrite(res3, 'Mean.png');
res4 = medfilt2(img);
imwrite(res4, 'Median.png')