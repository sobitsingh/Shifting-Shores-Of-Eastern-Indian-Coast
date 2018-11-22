picture=imread('Apr 1985.jpg');
imshow(picture);
BW=rgb2gray(picture);

imshow(BW);
e3=edge(BW,'Roberts');
imshow(e3);
j=imcrop(e3,[510 180 245 245]);

figure;
imshow(e3);
e4=edge(e3,'Sobel');
imshow(e4);


picture1=imread('Apr 1992.jpg');
imshow(picture1);
BW1=rgb2gray(picture1);

imshow(BW1);
e31=edge(BW1,'Roberts');
imshow(e31);
j1=imcrop(e31,[510 180 245 245]);


imshow(e31);
e41=edge(e31,'Sobel');
imshow(e41);

D1 = bwdist(e41,'euclidean'); 

subplot(2,2,1), imshow(D1), title('Euclidean_2');
D = bwdist(e4,'euclidean'); 

subplot(2,2,1), imshow(D), title('Euclidean');



p=D<1.005;

imshow(p),title('First');


y=D1<1.005;

imshow(y),title('Second');

   [m,n]=size(p);
  [~,idx]=max(fliplr(p),[],2);
newImage=  ( (n+1-idx)==(1:n) );
figure,imshow(newImage),title('First coastline');

[m1,n1]=size(y);
  [~,idx1]=max(fliplr(y),[],2);
newImage1=  ( (n1+1-idx1)==(1:n1) );
figure,imshow(newImage1),title('Second Coastline');
figure,imshow(newImage1),title('Second Coastline');


p=(1:size(newImage,2)).';
distance = double(newImage-newImage1)*p;
real_distance=abs(distance);
imshow(newImage-newImage1),title('Change in coastline');

