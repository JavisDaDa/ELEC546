I = imread('chessboard.jpg');
rotateI = imrotate(I, 30, 'nearest');
imshow(I);
imshow(rotateI);
imwrite(rotateI, 'rotate30.jpg')
S=imresize(I,4,'nearest');
imwrite(S, 'resize4.jpg')
imshow(S)