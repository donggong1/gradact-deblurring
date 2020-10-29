function [x_out, k] = ss_blind_deconv(y_in, x_in, k, x_mpl_opts, k_est_opts, opts)
%% initilize scale size
ks = size(k);
hks = floor(ks./2);
if(ks(1)==ks(2))
    hks_h = hks(1);
    hks_w = hks_h;
else
    hks_h = hks(1);
    hks_w = hks(2);
end
num_fd = length(y_in);

xk_maxIter = opts.xk_maxIter;
costlist = zeros(1, xk_maxIter+1);

x_est = x_in;
y = y_in;

lambda_k_dec_rate = 1;

%% parameters
for xk_iter = 1:xk_maxIter
    %     lambda_k = max(k_est_opts.lambda_k, 100*0.5^xk_iter);
    fprintf('--x-k iter: %d / %d\n', xk_iter, xk_maxIter);
    %% ESTIMATION OF IMAGE (X)
    mpl_gpl_lambda_x = x_mpl_opts.lambda_x;
    mpl_gpl_supp_init_rate = x_mpl_opts.supp_init_rate;
    mpl_gpl_map_prun_size = x_mpl_opts.map_prun_size;
    mpl_gpl_max_mpl_num = x_mpl_opts.max_mpl_num;
    mpl_gpl_min_mpl_num = x_mpl_opts.min_mpl_num;
    mpl_gpl_stop_relative_rate_x = x_mpl_opts.stop_relative_rate_x;
    mpl_gpl_map_ext_rate = x_mpl_opts.map_ext_rate;
    mpl_gpl_inner_max_ite = x_mpl_opts.inner_max_ite;
    mpl_gpl_inner_stop_relative_rate = x_mpl_opts.inner_stop_relative_rate;
    mpl_gpl_is_display_x = x_mpl_opts.is_display_x;
    [x_est, supp_map, objvalue_list_list_x] ...
        = solve_x_actgrad_apg(y, x_est, k, mpl_gpl_lambda_x, mpl_gpl_supp_init_rate,...
        mpl_gpl_map_prun_size, mpl_gpl_max_mpl_num, mpl_gpl_min_mpl_num, ...
        mpl_gpl_stop_relative_rate_x, mpl_gpl_map_ext_rate, ...
        mpl_gpl_inner_max_ite, mpl_gpl_inner_stop_relative_rate, mpl_gpl_is_display_x);
    %% update x and y
    inner_cut_rate = 2;
    for i= 1:num_fd
        tmp = y{i};
        y_inner{i} = tmp(hks_h*inner_cut_rate+1:end-hks_h*inner_cut_rate,...
            hks_w*inner_cut_rate+1:end-hks_w*inner_cut_rate);
    end
    for i= 1:num_fd
        x_inner{i} = x_est{i}(hks_h*(inner_cut_rate-1)+1:end-hks_h*(inner_cut_rate-1),...
            hks_w*(inner_cut_rate-1)+1:end-hks_w*(inner_cut_rate-1));
    end
    
    %% ESTIMATION OF KERNEL (K)
    %% lambda_k decreasing rate
    if(opts.is_k_lambda_dec)
        lambda_k_dec_rate = min(lambda_k_dec_rate*1.068, 2);
    end
    %%
    k_prev = k;
    switch k_est_opts.type
        case 'l2_conj'
            opts_k_l2.use_fft = 1;
            opts_k_l2.lambda = k_est_opts.lambda_k/lambda_k_dec_rate;
            opts_k_l2.pcg_tol = k_est_opts.tol;%1e-4;
            opts_k_l2.pcg_its = k_est_opts.l2_conj_iter_max; %2
            k = solve_k_l2_conj(k_prev, x_inner, y_inner, opts_k_l2);
            objvalue_list_k = 0;
            ks_list_k = 0;
        case 'irls'
            opts_k_irls.use_fft = 1;
            opts_k_irls.lambda = k_est_opts.lambda_k/lambda_k_dec_rate;
            opts_k_irls.pcg_tol = k_est_opts.tol;%1e-4;
            opts_k_irls.pcg_its = k_est_opts.l2conj_iter_max;
            k = solve_k_irls(k_prev, x_inner, y_inner, opts_k_irls);
            objvalue_list_k = 0;
            ks_list_k = 0;
        otherwise
            fprintf('Unkown typd.\n');
    end
    %% post peocessing for l2 and irls
    if(strcmp(k_est_opts.type, 'l2_conj')||strcmp(k_est_opts.type, 'irls'))
        k(k < max(k(:))*opts.kernel_inner_pruning_rate) = 0; % 0.05
        k = k./sum(k(:));
        CC = bwconncomp(k,8);
        for ii=1:CC.NumObjects
            currsum=sum(k(CC.PixelIdxList{ii}));
            if(currsum< 0.02)
                k(CC.PixelIdxList{ii}) = 0;
            end
        end
        k(k<0) = 0; k=k/sum(k(:));
    end
    %% display
    if(opts.is_display)
        figure(111);
        subplot(2,4,1); plot(objvalue_list_list_x{1}); title('x_h');
        subplot(2,4,2); plot(objvalue_list_list_x{2}); title('x_v');
        subplot(2,4,3); plot(objvalue_list_k); title('k');
        subplot(2,4,4); plot(ks_list_k); title('k error');   
        subplot(2,4,5); imshow(abs(x_est{1}), []); title(['xk iter', num2str(xk_iter)]);
        subplot(2,4,6); imshow(abs(x_est{2}), []);
        subplot(2,4,7); imshow(k, []);
%         subplot(2,4,8); imshow(k, []);
    end
end
x_out = x_est;

return