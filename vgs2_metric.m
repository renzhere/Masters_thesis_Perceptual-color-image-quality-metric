%VGS=Visual Gradient Similarity

function [VGS, VGS2] = vgs2_metric(ref_image, dist_image)

I= imread(ref_image);
R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);

I = (0.2989 * R) + (0.587 * G) + (0.114 * B);
IR = im2double(I);
IR = image_Crop(IR);

T0 = [1/4 1/4;1/4 1/4];
IRo = conv2( IR,T0,'same');

% level 1 is original image----IRo
[mag_orig_0, dir_orig_0] = gradient_md(IRo);

% figure, subplot(1,2,1),imshow(mag_orig_0,[]);title('G magnitude');
% subplot(1,2,2),imshow(dir_orig_0,[]);title('G direction');

% scaling level 1
IR1= scaling(IRo);
[mag_o1, dir_o1] = gradient_md(IR1);

% scaling level 2
IR2 = scaling(IR1);
[mag_o2, dir_o2] = gradient_md(IR2);

I1  = imread(dist_image);
R1  = I1(:,:,1);
G1  = I1(:,:,2);
B1  = I1(:,:,3);

I1  = (0.2989 * R1) + (0.587 * G1) + (0.114 * B1);
ID  = im2double(I1);
ID  = image_Crop(ID);
T0  = [1/4 1/4;1/4 1/4];
IDo = conv2( ID,T0,'same');


% level 1 is original image----IDo
[mag_t0, dir_t0] = gradient_md(IDo);


figure,subplot(1,2,1), imshow(mag_orig_0,[]);title('Gradient map of Original image');
subplot(1,2,2), imshow(mag_t0,[]),title('Gradient map of Distorted Image');
% figure, subplot(1,2,1),imshow(mag_t0,[]);title('G magnitude');
% subplot(1,2,2),imshow(dir_t0,[]);title('G direction');

% scaling level 1
ID1= scaling(IDo);
[mag_t1, dir_t1] = gradient_md(ID1);

% scaling level 2
ID2= scaling(ID1);
[mag_t2, dir_t2] = gradient_md(ID2);



% GRADIENT VECTOR MAP
% obtain the three scales gradient vector map of the original image

mapo0=gradient_vector(mag_orig_0, dir_orig_0);      % scale 0
mapo1=gradient_vector(mag_o1, dir_o1);              % scale 1
mapo2=gradient_vector(mag_o2, dir_o2);              % scale 2

% obtain the three scales gradient vector map of the distorted image

mapt0=gradient_vector(mag_t0, dir_t0);              % scale 0
mapt1=gradient_vector(mag_t1, dir_t1);              % scale 1
mapt2=gradient_vector(mag_t2, dir_t2);              % scale 2

% ANGULAR DIFFERENCES

[ a_0] = ang_diff(mapo0, mapt0, mag_orig_0, dir_orig_0, mag_t0, dir_t0);    % scale 0
[ a_1] = ang_diff(mapo1, mapt1, mag_o1, dir_o1, mag_t1, dir_t1);            % scale 1
[ a_2] = ang_diff(mapo2, mapt2, mag_o2, dir_o2, mag_t2, dir_t2);            % scale 2

%  to obtain the threshold some constants
ep = 3; % scaling factor

% for dislay type CRT/LCD
r  = 0.50;
T  = 0.030;

% % the default parameter values for VGS are set to be
% r = 0.55;
% T = 0.025;

% % for dislay type PRINTING
% r  = 0.85;
% T  = 0.035;
thresh = ep * T * 255;

% projection coefficient
 [lambda_0,Ns_0,i0,j0] = projection_coeff(thresh,mapo0,mapt0);
 [lambda_1,Ns_1,i1,j1] = projection_coeff(thresh,mapo1,mapt1);
 [lambda_2,Ns_2,i2,j2] = projection_coeff(thresh,mapo2,mapt2);
 

%  the global contrast of the test image with that of the reference image
%  is obtained by multiplyuing lambda with mo_r
g_contrast_0 = lambda_0 * (mapo0.^r);
g_contrast_1 = lambda_1 * (mapo1.^r);
g_contrast_2 = lambda_2 * (mapo2.^r);

[phi_s0 , sci_0]= psgm(g_contrast_0,mapt0.^r);
[phi_s1 , sci_1]= psgm(g_contrast_1,mapt1.^r);
[phi_s2 , sci_2]= psgm(g_contrast_2,mapt2.^r);


% the pointwise similarity of gradient magnitude between the test image and
% the reference image

% is obtained by projecting the smaller one onto the larger one;
c_0 = phi_s0 ./ sci_0;
c_1 = phi_s1 ./ sci_1;
c_2 = phi_s2 ./ sci_2;


% The pointwise gradient similarity
qs_0 = a_0 .* c_0;
qs_1 = a_1 .* c_1;
qs_2 = a_2 .* c_2;

figure, imshow(qs_0);title('Pointwise Gradient Similarity map')

% the similarity map
% figure, subplot(1,3,1),imshow(qs_0,[]);
% subplot(1,3,2),imshow(qs_1,[]);
% subplot(1,3,3),imshow(qs_2,[]);

qs_0 = remov_NAN(qs_0);
qs_1 = remov_NAN(qs_1);
qs_2 = remov_NAN(qs_2);

Qs_0 = intrascale_pooling(qs_0,Ns_0,i0,j0);
Qs_1 = intrascale_pooling(qs_1,Ns_1,i1,j1);
Qs_2 = intrascale_pooling(qs_2,Ns_2,i2,j2);

mu0  = quality_uniformity(qs_0,Qs_0,i0,j0,Ns_0);
mu1  = quality_uniformity(qs_1,Qs_1,i1,j1,Ns_1);
mu2  = quality_uniformity(qs_2,Qs_2,i2,j2,Ns_2);

Qs_a0 = mu0 * Qs_0;
Qs_a1 = mu1 * Qs_1;
Qs_a2 = mu2 * Qs_2;

Qs_b0 = lambda_0 * Qs_a0;
Qs_b1 = lambda_1 * Qs_a1;
Qs_b2 = lambda_2 * Qs_a2;

% VGS1
VGS = (Qs_b0 + Qs_b1 + Qs_b2)/3;


% VGS2
tau = 2;
k   = 2;
% f_peak  = 4;
fb  = 1;
% lv  = 14 * 128 ;
% l * v = 2.5 * 512 /0.8;
% ws  = 0.26 * ( 0.0192 + 0.114 * k * f )^ (-(0.114 * k * f) ^ 1.1);
% V   = lv * tan( pi/180);
% fb  = (2 ^ (-5) * V) ;

fs_0  = ((2 ^ (5-1)) * fb) / tau;
ws_0  = 0.26 * ( 0.0192 + 0.114 * k * fs_0 ) * exp (-(0.114 * k * fs_0) ^ 1.1);

fs_1  = ((2 ^ (5-2)) * fb) / tau;
ws_1  = 0.26 * ( 0.0192 + 0.114 * k * fs_1 ) * exp (-(0.114 * k * fs_1) ^ 1.1);

fs_2  = ((2 ^ (5-3)) * fb) / tau;
ws_2  = 0.26 * ( 0.0192 + 0.114 * k * fs_2 ) * exp (-(0.114 * k * fs_2) ^ 1.1);

% Qs_b02=lambda_0*Qs_a0;
% Qs_b12=lambda_1*Qs_a1;
% Qs_b22=lambda_2*Qs_a2;

% S1= 1;
% S2= ceil(log2(2 * V)/(tau ));

VGS2 = ((Qs_b0 * ws_0) + (Qs_b1 * ws_1) + (Qs_b2 * ws_2))/(ws_0 + ws_1 + ws_2);
% export2base();

end



function [scaled_image]=scaling(image)
T0 = [1/4 1/4;1/4 1/4];
Ig = conv2( image,T0,'same');
I1 = downsample(Ig,2);
I2 = downsample(I1',2)';
scaled_image=I2;
end

function [lambda , Ns, i1,j1]= projection_coeff(thresh,mapo,mapt)

[r1,c1] = size(mapo);
r  = 0.5;

% raw gradient magnitude m:
mapo_r=(mapo.^r);
mapt_r=(mapt.^r);

i1 = [];
j1 = [];

n=0;
for i= 1:r1
    for j= 1:c1
        if (mapo(i,j) >= thresh) || (mapt(i,j) >= thresh)
            n=n+1;
            i1(n)=i;
            j1(n)=j;
         end
    end
end
save('ival.mat','i1');
save('jval.mat','j1');

Ns = n;

load('ival.mat', 'i1');
load('jval.mat', 'j1');

[ci1]      =    size(i1,2);
[mo, no]   =    size(mapo_r);

 num_map=0;
 den_map=0;
 in = 0;

for i=1:mo
    for j= 1:no
        if(in<ci1)
            in=in + 1;
%             mapo_r_thresh(in)=mapo_r(i1(1,in),j1(1,in));
%             mapt_r_thresh(in)=mapt_r(i1(1,in),j1(1,in));
            num_map1 = num_map + (mapo_r(i1(1,in),j1(1,in)) * mapt_r(i1(1,in),j1(1,in)));
            den_map1 = den_map + (mapo_r(i1(1,in),j1(1,in)) * mapo_r(i1(1,in),j1(1,in)));
            
        end
    end
end
% img_o=mapo_r_thresh(in);
% img_t=mapt_r_thresh(in);
lambda = num_map1/den_map1; 

end

function [phi,sci]= psgm(img1,img2)

[r, c]=size(img1);

phi=zeros(r,c);
sci=zeros(r,c);

for i=1:r
    for j=1:c
        a=img1(i,j);
        b=img2(i,j);
        
        if (a>b)
            phi(i,j)=b;
            sci(i,j)=a;
        else
            phi(i,j)=a;
            sci(i,j)=b;
        end
    end
end

end

function [Qs]= intrascale_pooling(qs,N,I,J)
i1 = I;
j1 = J;
[ci1] = size(i1,2);

[mo, no]   = size(qs);
in      = 0;
qs_sum  = 0;
 
for i=1:mo
    for j= 1:no
        if(in<ci1)
            in=in + 1;
            qs_sum = qs_sum + qs(i1(1,in),j1(1,in));
        end
    end
end
% qs_sum;
% S=sum(sum(ac));
Qs=qs_sum/N;
end

function [mu]= quality_uniformity(qs,Q,I,J,N)

i1 = I;
j1 = J;
[ci1] = size(i1,2);
% [rj1, cj1] = size(j1)

[mo, no]   = size(qs);

in   = 0;
sum  = 0;
 
for i=1:mo
    for j= 1:no
        if(in<ci1)
            in=in + 1;
            H=qs(i1(1,in),j1(1,in));
            PP=(abs(H - Q)^2);
            sum = sum + PP;
        end
    end
end
% S=sum(sum(ac));
S1=sum/N;
S2=sqrt(S1);

mu=1-S2;

end

function [map]=gradient_vector(m_image, d_image)
Gmag = m_image;
Gdir = d_image;
[r,c]=size(Gmag);
ref_array  = zeros(r,c);

for i=1:r
    for j=1:c
        ref_array(i,j)= sqrt((Gmag(i,j) * Gmag(i,j)) + (Gdir(i,j) * Gdir(i,j)));
    end
end
map = ref_array;
end

function [a] = ang_diff(mo_mag, mt_mag, Gmag_o,  Gdir_o, Gmag_t, Gdir_t)
% orig image 
no = Gmag_o;    % magnitude
eo = Gdir_o;    % direction

% test image 
nt = Gmag_t;    % magnitude
et = Gdir_t;    % direction

mo = mo_mag;    % grad vector map, orig image
mt = mt_mag;    % grad vector map, test image

[r,c] = size(mo);
pdt   = zeros(r,c);

cos_D_theta  = zeros(r,c);
grad_dir_sim = zeros(r,c);

for i = 1:r
    for j = 1:c
        pdt(i,j)          =  (no(i,j) * nt(i,j)) + (eo(i,j) * et(i,j));
        cos_D_theta(i,j)  =  pdt(i,j) / (mo(i,j) * mt(i,j));
        grad_dir_sim(i,j) =  abs(cos_D_theta(i,j)) * cos_D_theta(i,j);
    end
end

a  = remov_NAN(grad_dir_sim);


end
function [im] = image_Crop(image)
% I=imread('cactus.png');figure, imshow(I);
% im= image(111:238,77: 204);
% im= image(10:137,10:137);
% im= image(10:41,10:41);
% imwrite(im,'bird.png');
% figure, imshow(im);
 im=image;
end

function [mag, dir] = gradient_md(I)

% display('original dimension= ');
% display(size(I));

[Gmag, Gdir] = imgradient(I,'prewitt');

mag = Gmag;
dir = Gdir;

end

function[a] = remov_NAN(a)

 ind_plain           = find(isnan(a));
[row_indx, col_indx] = ind2sub(size(a), ind_plain);

    for el_id = 1:length(row_indx)
        a(row_indx(el_id),col_indx(el_id)) =  1;
    end

end



