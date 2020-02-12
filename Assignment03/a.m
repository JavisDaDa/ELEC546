%a
filter1=[2,4,5,4,2;
    4,9,12,9,4;
    5,12,15,12,5;
    4,9,12,9,2;
    2,4,5,4,2];
filter1=filter1/159;
img=imread('cameraman.tif');
res=imfilter(img,filter1,'symmetric');
imshow(res)
imwrite(res,'denoisedman.tif')
%b
filter2= [-1,0,1;-2,0,2;-1,0,1];
filter3= [1,2,1;0,0,0;-1,-2,-1];
res1= imfilter(res,filter2);
imwrite(res1,'Dx.tif')
res2= imfilter(res,filter3);
imwrite(res2,'Dy.tif')
d= sqrt(double(res1.^2 + res2.^2));
d(:,1)=0;
chu = double(res2)./double(res1);
theta=zeros(256,256,'double');
theta(find(chu<=tan(pi/8) & chu>=-tan(pi/8)))=0;
theta(find(chu==Inf))= pi/2;
theta(find(tan(pi/8)<=chu & chu<=tan(3*pi/8)))=pi/4;
theta(find(chu>=tan(3*pi/8) & chu<=-tan(3*pi/8)))=pi/2;
theta(find(chu>=-tan(3*pi/8) & chu<=-tan(pi/8)))=3*pi/4;
imshow(theta)