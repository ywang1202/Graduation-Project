clc;
clear;
close all;
warning off;
addpath(genpath(cd));

% I=double(imread('lake_IR.png'))/255;
% V=double(imread('lake_VIS.png'))/255;

calc_metric = 0; % Calculate the metrices is time consuming, it is used for quantitative evaluation. Set it to 0 if you do not want to do it.

for n = 1:20
    I = double(imread(strcat('~/毕设/Code/IV_images/IR', num2str(n, '%02d'), '.png')))/255;
    V = double(imread(strcat('~/毕设/Code/IV_images/VIS', num2str(n, '%02d'), '.png')))/255;
    %The proposed GTF
    nmpdef;
    pars_irn = irntvInputPars('l1tv');

    pars_irn.adapt_epsR   = 1;
    pars_irn.epsR_cutoff  = 0.01;   % This is the percentage cutoff
    pars_irn.adapt_epsF   = 1;
    pars_irn.epsF_cutoff  = 0.05;   % This is the percentage cutoff
    pars_irn.pcgtol_ini = 1e-4;
    pars_irn.loops      = 5;
    pars_irn.U0         = I-V;
    pars_irn.variant       = NMP_TV_SUBSTITUTION;
    pars_irn.weight_scheme = NMP_WEIGHTS_THRESHOLD;
    pars_irn.pcgtol_ini    = 1e-2;
    pars_irn.adaptPCGtol   = 1;

    tic;
    U = irntv(I-V, {}, 4, pars_irn);
    t0=toc;

    X=U+V;
    X=im2gray(X);
    imwrite(X, strcat('./outputs/GTF_fuse_', num2str(n, '%02d'), '.png'));
    if calc_metric, Result = Metric(uint8(abs(I)*255),uint8(abs(V)*255),uint8(abs(X*255))); end
end