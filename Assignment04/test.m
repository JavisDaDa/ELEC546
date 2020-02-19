I = imread('chessboard.jpg');

figure; imshow(I);

forward_dx = mipforwarddiff(I,'dx'); figure, imshow(forward_dx);

forward_dy = mipforwarddiff(I,'dy'); figure, imshow(forward_dy);

central_dx = mipcentraldiff(I,'dx'); figure, imshow(central_dx);

central_dy = mipcentraldiff(I,'dy'); figure, imshow(central_dy);
