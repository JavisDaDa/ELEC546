  
result=res;
for i = 2:255
    for j = 2:255
        if theta(i,j)==0
            if res(i,j)>res(i+1,j) && res(i,j)>res(i-1,j) 
                result(i,j)=255;
            else
                result(i,j)=0;
            end
        end
        if theta(i,j)==pi/2
            if res(i,j)>res(i,j+1) && res(i,j)>res(i,j-1)
                result(i,j)=255;
            else
                result(i,j)=0;
            end
        end
         if theta(i,j)==pi/4 
            if res(i,j)>res(i+1,j+1) && res(i,j)>res(i-1,j-1) 
                result(i,j)=255;
            else
                result(i,j)=0;
            end
         end
        if theta(i,j)==3*pi/4 
            if res(i,j)>res(i-1,j+1) && res(i,j)>res(i+1,j-1) 
                result(i,j)=255;
            else
                result(i,j)=0;
            end
        end
    end
end
result(result<200)=0;
result(:,1)=0;
imshow(result)
imwrite(result,'Part_c.jpg')
result1=result;