% Andrew Radwan
% 7.1.2026
% GPU Accelerated Computing
%
% EBL test 2

clear
close all

% % % % % % % % % % % % % % % % % % % % % % % % %
% Convolution using the data directly from the
% reference paper
% % % % % % % % % % % % % % % % % % % % % % % % %


% power gaussian model data from the paper
figure(5)
P = readmatrix("powerGaussian.csv");
PGTime = P(:,1);
PG = P(:,2);
loglog(PGTime,PG)
title("Power Gaussian data from paper")

% create convolution mask based on the energy spread
% from the power gaussian data
figure(6)

maskSize = 65;
center = (maskSize + 1) / 2;

% radial distance of every mask pixel from the center (same units as PGTime)
[x, y] = meshgrid(1:maskSize, 1:maskSize);
radii = sqrt((x - center).^2 + (y - center).^2);

% interpolate the measured profile directly onto those radii
% (radius 0 -> PGTime(1), your peak value; beyond max(PGTime) -> 0)
mask = interp1(PGTime, PG, radii, 'linear',0);
mask(33,33) = 1;

% normalize so the center pixel is exactly 1.0
mask = mask / mask(center, center);

imagesc(mask)
title("Convolution mask")


% Input IC
figure(7)
IC = imread("ICtest1.png");
IC = IC(:,:,1);
imshow(IC)
title("IC Layout")

% Exposed IC
figure(8)
IC2 = conv2(IC,mask,"same");
imagesc(IC2)
title("Convolution")