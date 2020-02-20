I = imread('chessboard.jpg');
rotateI = imrotate(I, 30, 'nearest');
imshow(I);
imshow(rotateI);
