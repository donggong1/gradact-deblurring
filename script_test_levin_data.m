% test on the noise free Levin's data
clear
close all
addpath('utils')
addpath('evaluation');
addpath('sparse_nbd');

root_path = '/your/workdir/';
inpath = [root_path, '/datasets/levin_data/'];
outpath = [root_path, '/results/gradact/'];

if(~isdir(outpath))
    mkdir(outpath)
end

is_save_mat = logical(1);
is_save_img = logical(1);

%% basic setting
prescale = 1;
opts.kernel_est_win = [];
opts.gamma_correct = 1;
opts.kernel_size = 31;

opts.is_display = logical(1);
opts.xk_maxIter = 15;

x_est_opts.lambda_x = 5e-3;
x_est_opts.lambda_x_final = 1e-4;
x_est_opts.supp_init_rate = 0.5;
x_est_opts.map_prun_size = 4;
x_est_opts.max_act_num = 10; 
x_est_opts.min_act_num = 4; 
x_est_opts.stop_relative_rate_x = 1e-5;
x_est_opts.map_ext_rate = 1;
x_est_opts.inner_max_ite = 20;
x_est_opts.inner_stop_relative_rate = 1e-6;
x_est_opts.is_display_x = logical(0);

k_est_opts.lambda_k = 0.01;
k_est_opts.tol = 1e-4;
k_est_opts.l2_conj_iter_max = 10;

%%
opts.k_final_iso_prun = logical(1);
opts.k_thresh = 20;
opts.kernel_inner_pruning_rate = 0.02;
% opts.is_kernel_diff = logical(0);
opts.noise_smooth = [2, 0];

%% options for non-blind deconvolution
% opts.nonblind_type = 'hyper_Laplacian'; opts.nb_lambda = 3000; opts.nb_alpha = 0.5; opts.use_ycbcr = 1;
opts.nonblind_type = 'sparse_Levin'; opts.levin_lambda = 0.001; opts.levin_ite = 100;
% opts.nonblind_type = 'Noisy_Zhong'; opts.zl_reg=1;
% opts.nonblind_type = 'EPLL'; opts.epll_noise_level = 0.01; opts.epll_patch_size = 8;%opts.=1;
% excludeList = []; opts.epll_prior = @(Z,patchSize,noiseSD,imsize) aprxMAPGMM(Z,patchSize,noiseSD,imsize,GS,excludeList);
% opts.LogLFunc = [];

%% ------------------------
%% running on data
kernel_size_list = [19, 17, 15, 27, 13, 21, 23, 23];
for iter_x = 5:8
    for iter_k = 1:8
        fprintf('image: %d, kernel: %d\n', iter_x, iter_k);
        opts.kernel_size = kernel_size_list(iter_k); idx_k = iter_k;
        load([inpath, 'im0', num2str(iter_x), '_flit0', num2str(iter_k), '.mat']); idx_x = iter_x;
        y_input = y;  gt_x = x;  gt_k = rot90(rot90(f));
        clear x y f 
        [est_k, est_x_grad, time_k_estimation] ...
            = ms_blind_devonv(y_input, x_est_opts, k_est_opts, opts);
        %%
        kk = rot90(rot90(est_k));
        for c= 1:size(y_input,3)
            deblur_img(:,:,c) = deconvSps(y_input(:,:,c), kk, opts.levin_lambda, opts.levin_ite); 
        end
        est_x = deblur_img;
        %% Evaluation kernel and deblurred iamge
        ks = size(est_k,1);
        ks = floor(ks/2)+1;
        [ssde, ~] = ssd_cal(est_x(1+ks:end-ks, 1+ks:end-ks), gt_x(1+ks:end-ks, 1+ks:end-ks));
        i = iter_x-4; j = iter_k;
        fprintf('im=%d, k=%d, ssde=%f\n', i, j, ssde);
        %% 
        if(is_save_mat)
            if(~isdir([outpath, '/Mat/']))
                mkdir([outpath, '/Mat/'])
            end
            save([outpath, '/Mat/im0', num2str(i), '_k0',num2str(j)],...
                'y_input', 'gt_x','gt_k','est_x','est_k','ssde');
        end
        if(is_save_img)
            if(~isdir([outpath, '/Image/']))
                mkdir([outpath, '/Image/'])
            end
            imwrite(uint8(est_x*255), [outpath, ...
                '/Image/im0', num2str(i), '_k0',num2str(j),'_image','.jpg']);
            out_k = est_k - min(est_k(:)); out_k = out_k./max(out_k(:));
            imwrite(uint8(out_k*255), [outpath, ...
                '/Image/im0', num2str(i), '_k0',num2str(j),'_kernel','.png']);
        end
    end
end

        
        