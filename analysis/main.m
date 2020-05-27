% Li H, Wu X J. DenseFuse: A Fusion Approach to Infrared and Visible Images[J]. arXiv preprint arXiv:1804.08361, 2018. 
% https://arxiv.org/abs/1804.08361

% fileName_source_ir  = [""];
% fileName_source_vis = [""];
% fileName_fused      = [""];
fuse_method = "our";
disp("Start");
epoch = 1;
for n=1:20
    
    source_image1 = imread(strcat("~/毕设/Code/IV_images/IR", num2str(n, "%02d"), ".png"));
    source_image2 = imread(strcat("~/毕设/Code/IV_images/VIS", num2str(n, "%02d"), ".png"));
    fused_image   = imread(strcat("~/毕设/Code/3/", fuse_method, "/outputs/add/", num2str(epoch), "/", fuse_method, "_fuse_", num2str(n, "%02d"), ".png"));
    
    [EN(n), MI(n), Qabf(n), FMI_pixel(n), FMI_dct(n), FMI_w(n), SCD(n), SSIM(n)] = analysis_Reference(fused_image,source_image1,source_image2);
    disp([EN(n), MI(n), Qabf(n), FMI_pixel(n), FMI_dct(n), FMI_w(n), SCD(n), SSIM(n)]);
end

xlswrite(strcat("./", fuse_method, "add", num2str(epoch), ".csv"), [EN; MI; Qabf; FMI_pixel; FMI_dct; FMI_w; SCD; SSIM]);
% end
disp('Done');