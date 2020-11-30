function [x_out, k] = ss_blind_deconv(y_in, x_in, k, x_est_opts, k_est_opts, opts)
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
%%
for xk_iter = 1:xk_maxIter
    %     lambda_k = max(k_est_opts.lambda_k, 100*0.5^xk_iter);
    fprintf('--x-k iter: %d / %d\n', xk_iter, xk_maxIter);
    %% ESTIMATION OF IMAGE (X)
    estx_act_lambda_x = x_est_opts.lambda_x;
    estx_act_supp_init_rate = x_est_opts.supp_init_rate;
    estx_act_map_prun_size = x_est_opts.map_prun_size;
    estx_act_max_act_num = x_est_opts.max_act_num;
    estx_act_min_act_num = x_est_opts.min_act_num;
    estx_act_stop_relative_rate_x = x_est_opts.stop_relative_rate_x;
    estx_act_map_ext_rate = x_est_opts.map_ext_rate;
    estx_act_inner_max_ite = x_est_opts.inner_max_ite;
    estx_act_inner_stop_relative_rate = x_est_opts.inner_stop_relative_rate;
    estx_act_is_display_x = x_est_opts.is_display_x;
    [x_est, supp_map, objvalue_list_list_x] ...
        = solve_x_actgrad_apg(y, x_est, k, estx_act_lambda_x, estx_act_supp_init_rate,...
        estx_act_map_prun_size, estx_act_max_act_num, estx_act_min_act_num, ...
        estx_act_stop_relative_rate_x, estx_act_map_ext_rate, ...
        estx_act_inner_max_ite, estx_act_inner_stop_relative_rate, estx_act_is_display_x);
    %% update x and y
    boundary_cut_rate = 2;
    for i= 1:num_fd
        tmp = y{i};
        y_inner{i} = tmp(hks_h*boundary_cut_rate+1:end-hks_h*boundary_cut_rate,...
            hks_w*boundary_cut_rate+1:end-hks_w*boundary_cut_rate);
    end
    for i= 1:num_fd
        x_inner{i} = x_est{i}(hks_h*(boundary_cut_rate-1)+1:end-hks_h*(boundary_cut_rate-1),...
            hks_w*(boundary_cut_rate-1)+1:end-hks_w*(boundary_cut_rate-1));
    end
    %% ESTIMATION OF KERNEL (K)
    %% lambda_k decreasing rate
    if(opts.is_k_lambda_dec)
        lambda_k_dec_rate = min(lambda_k_dec_rate*1.068, 2);
    end
    %%
    k_prev = k;
    opts_k_l2.use_fft = 1;
    opts_k_l2.lambda = k_est_opts.lambda_k/lambda_k_dec_rate;
    opts_k_l2.pcg_tol = k_est_opts.tol; 
    opts_k_l2.pcg_its = k_est_opts.l2_conj_iter_max; 
    k = solve_k_l2_conj(k_prev, x_inner, y_inner, opts_k_l2);
    objvalue_list_k = 0;
    ks_list_k = 0;
    %% k post peocessing
    k(k < max(k(:))*opts.kernel_inner_pruning_rate) = 0; 
    k = k./sum(k(:));
    CC = bwconncomp(k,8);
    for ii=1:CC.NumObjects
        currsum=sum(k(CC.PixelIdxList{ii}));
        if(currsum< 0.02)
            k(CC.PixelIdxList{ii}) = 0;
        end
    end
    k(k<0) = 0; k=k/sum(k(:));
    %% display
    if(opts.is_display)
        figure(1024);
        subplot(2,4,1); plot(objvalue_list_list_x{1}); title('x_h');
        subplot(2,4,2); plot(objvalue_list_list_x{2}); title('x_v');
        subplot(2,4,3); plot(objvalue_list_k); title('k');
        subplot(2,4,4); plot(ks_list_k); title('k error');   
        subplot(2,4,5); imshow(abs(x_est{1}), []); title(['xk iter', num2str(xk_iter)]);
        subplot(2,4,6); imshow(abs(x_est{2}), []);
        subplot(2,4,7); imshow(k, []);
    end
end
x_out = x_est;

return