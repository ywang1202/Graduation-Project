clc;
clear;
close all;
warning off;
addpath(genpath(cd));

calc_metric = 0; % Calculate the metrices is time consuming, it is used for quantitative evaluation. Set it to 0 if you do not want to do it.

for n = 1:20
    I = double(imread(strcat('~/毕设/Code/IV_images/IR', num2str(n, '%02d'), '.png')))/255;
    V = double(imread(strcat('~/毕设/Code/IV_images/VIS', num2str(n, '%02d'), '.png')))/255;
    %The laplacian pyramid as a compact image code(1983)
    level=4;
    tic;
    X = lp_fuse(I, V, level, 3, 3);       %LP
    toc;
    X=im2gray(X);
    imwrite(X, strcat('./outputs/LP_fuse_', num2str(n, '%02d'), '.png'));
    if calc_metric, Result1 = Metric(uint8(abs(I)*255),uint8(abs(V)*255),uint8(abs(X1*255))); end
end