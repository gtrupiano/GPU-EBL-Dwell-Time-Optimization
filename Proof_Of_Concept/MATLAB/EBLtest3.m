% Andrew Radwan
% 7.7.2026
% GPU Accelerated Computing
%
% EBL test 3

clear
close all

% % % % % % % % % % % % % % % % % % % % % % % % %
% Convolution using the data directly from the
% reference paper
% % % % % % % % % % % % % % % % % % % % % % % % %


% power gaussian model data from the paper
f = 1;
figure(f)
f = f + 1;
P = readmatrix("powerGaussian.csv");
PGTime = P(:,1);
PG = P(:,2);
loglog(PGTime,PG)
title("Power Gaussian data from paper")

% create convolution mask based on the energy spread
% from the power gaussian data
figure(f)
f = f + 1;
maskSize = 65;
center = (maskSize + 1) / 2;

% radial distance of every mask pixel from the center (same units as PGTime)
[x, y] = meshgrid(1:maskSize, 1:maskSize);
radii = sqrt((x - center).^2 + (y - center).^2);

% interpolate the measured profile directly onto those radii
% (radius 0 -> PGTime(1), your peak value; beyond max(PGTime) -> 0)
mask = interp1(PGTime, PG, radii, 'linear',0);

% mask(33,33) = 8.681604556000000e+06;
% % % % % This is cheating, I care more about computations
mask = mask./2; % Reducing the spread
mask(33,33) = 1; % Making sure center is 1

% normalize so the center pixel is exactly 1.0
mask = mask / mask(center, center);

imagesc(mask)
title("Convolution mask")


% Input IC
figure(f)
f = f + 1;
IC = imread("ICtest1.png");
IC = IC(:,:,1);
imshow(IC)
title("IC Layout")

ICAdjusted = double(IC);

for index = 1:100
    if(mod(index, 10) == 0)
        figure(f)
        f = f + 1;
    end

    % Convolve
    ICExposed = conv2(ICAdjusted,mask,"same");
    subplot(2,2,1)
    imagesc(ICExposed)
    title("Convolution")


    % Get Error Matrix
    errorMatrix = ICExposed - double(IC);
    subplot(2,2,2)
    imagesc(errorMatrix)
    title("Error Matrix")
    
    % Calc Mean Squared Error
    errorMatrixNormalized = errorMatrix./max(errorMatrix, [], "all");
    mse(index) = mean(errorMatrixNormalized.^2, "all");
    subplot(2,2,4)
    bar(mse)
    title("Mean Squared Error")
    xlabel("Iteration")
    ylabel("MSE")

    % Adjust IC
    ICAdjusted = arrayfun(@updateIC, errorMatrix, ICAdjusted, rand(128)./10);
    % ICAdjusted = arrayfun(@updateIC2, errorMatrix, ICAdjusted);

    subplot(2,2,3)
    imshow(ICAdjusted, [0 255])
    title("IC Layout Adjusted")
end
