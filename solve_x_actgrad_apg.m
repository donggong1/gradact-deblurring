function [x_out, supp_map_out, objvalue_list_list] ...
    = solve_x_actgrad_apg(Y, X, k, lambda, supp_init_rate, map_prun_size,...
    max_act_num, min_act_num, stop_relative_rate, map_ext_rate,...
    inner_max_ite, inner_stop_relative_rate, is_display)

%% initialization
num_fd = length(X);
[h,w] = size(Y{1});
hks = size(k,1); wks = hks;
hhks = floor(hks/2); hwks = floor(wks/2);
x_out = cell(1, num_fd);
supp_map_out = cell(1, num_fd);
objvalue_list_list = cell(1,num_fd);

%% boundary mask
bound_mask = zeros(h,w);
bound_mask(1+hhks:end-hhks, 1+hwks:end-hwks) = 1;

%% seperately process different feature map
for iter_fd = 1:num_fd
    %%
    group_indicator_cell = cell(max_act_num);
    %%
    objvalue_list = zeros(max_act_num+1, 1);
    %% initialization
    y = Y{iter_fd};
    x = zeros(h,w);
    alpha = y; 
    %% set support number, and initlize support map
    g = conv2(y, rot90(k,2),'same');
    tmpg = -inf * ones(h,w);
    tmpg(1+hhks:end-hhks, 1+hwks:end-hwks) = abs(g(1+hhks:end-hhks, 1+hwks:end-hwks));
    
    tmplambdamax = max(tmpg(:));
    tmp = tmpg>=(supp_init_rate*tmplambdamax);
    slct_supp_num = sum(tmp(:));
    if(is_display)
        fprintf('slct_supp_num = %d\n', slct_supp_num);
    end
    %% support map
    vtmpg = tmpg(:); % note: tmpg is absolute value
    [~, tmpsidx] = sort(vtmpg,'descend');
    slct_vtmpidx = tmpsidx(1:slct_supp_num);
    tmpmap = zeros(h*w,1); tmpmap(slct_vtmpidx) = 1;
    supp_map = reshape(tmpmap, [h,w]);
    %%
    % support map refinement (pruning isolated small element group)
    % map_prun_size: small regions will be pruned
    supp_map = map_refine(supp_map, map_prun_size);
    group_indicator_cell{1} = supp_map;
    
    %% init obj value calculation
    tmpalpha = alpha.*bound_mask; 
    objvalue = 0.5*(tmpalpha(:)'*tmpalpha(:));
    objvalue_list(1) = objvalue;

    %% region selection and update inner elements
    for act_iter = 1:max_act_num
        %% update inner elements based on current support map
        % update the elements in map to decrease \xi=||y-Hx||_2^2
        x_in = x;
        % the output x is the masked version
        % solving the group lasso problem using APG 
        [x] = inner_x_solver_apg(x_in, y, k, supp_map, group_indicator_cell, lambda, inner_max_ite, inner_stop_relative_rate);
        %% update dual variable, reconstruct dual variable (alpha)
        alpha = y - conv2(x, k, 'same');
        %% update support map based on the residual (alpha)
        % select new region
        g = conv2(alpha, rot90(k,2), 'same');
        tmpg = -inf * ones(h,w);
        tmpg(1+hhks:end-hhks, 1+hwks:end-hwks) = ...
            abs(g(1+hhks:end-hhks, 1+hwks:end-hwks));
        vtmpg = tmpg(:);
        vtmpg(logical(supp_map(:))) = -inf; % excluding the selected region
        [~, tmpsidx] = sort(vtmpg,'descend');
        slct_vtmpidx = tmpsidx(1:slct_supp_num);
        %% merge new region and old region
        tmpmap = zeros(h*w,1); tmpmap(slct_vtmpidx) = 1;
        tmpmap = reshape(tmpmap, [h,w]);
        supp_map_prev = supp_map;
        supp_map = min(1, supp_map + tmpmap); % +: the merge operation
        %% support map refinement (pruning isolated small element group)
        supp_map = map_refine(supp_map, map_prun_size);
        tmp_supp = supp_map - supp_map_prev;
        group_indicator_cell{act_iter+1} = tmp_supp;
        %% check stop condition - TODO
        % update objective value
        r = alpha;
        tmpx = x.*supp_map; 
        tmpr = r.*bound_mask;
        objvalue = 0.5*(tmpr(:)'*tmpr(:)) + lambda*0.5*(tmpx(:)'*tmpx(:));
        objvalue_list(act_iter+1) = objvalue;
        
        if(is_display)
            fprintf('ite=%d, objvalue=%f ',act_iter, objvalue);
        end
                
        if(act_iter>=min_act_num)
            if(2*abs(objvalue - objvalue_list(act_iter))...
                    /abs(slct_supp_num*objvalue_list(1))<stop_relative_rate)
                fprintf('\n-- stop_ite:%d --\n', act_iter);
                break;
            end
        end
        if(is_display)
            fprintf('\n');
        end
    end
    x_out{iter_fd} = x;
    tmpmap = zeros(size(x));
    tmpmap(abs(x)>1e-10)=1;
    supp_map_out{iter_fd} = supp_map; 
    objvalue_list_list{iter_fd} =  objvalue_list;
    clear group_indicator_cell
end
return



%% x_out is masked version.
function [x_out] = inner_x_solver_apg(x_init, y, k, supp_map, group_indicator_cell, lambda, inner_max_ite, stop_relative_rate)
% update x based on APG only for elements in support map
x_est = x_init;
x_est_prev = x_est;
selc_map_ext = extend_map(supp_map, floor(size(k)/2), 0);
%% objective value calculation
r = y-conv2(x_est, k, 'same'); 
tmpr = r.*selc_map_ext; 
reg_val = 0;
for i=1:length(group_indicator_cell)
    tmpC = group_indicator_cell{i};
    if(isempty(tmpC))
        break;
    end
    tmp = x_est.*tmpC;
    reg_val = reg_val + norm(tmp(:));
end
f = 0.5*(tmpr(:)'*tmpr(:)); f_prev = f;
obj_crt = f + lambda*0.5*(reg_val^2); obj_prev = obj_crt;
obj0 = obj_crt;
%%
t = 1;
t_prev = t; 
s = 1; 
beta = 0.5;
max_inner_iter = inner_max_ite; 
%% optimize x in select region
for inner_iter=1:max_inner_iter
    v = x_est + ((t_prev-1)/t) * (x_est-x_est_prev);
    s_crt = s * beta; 
    grad = (conv2((conv2(v, k, 'same') - y), rot90(k, 2), 'same'));
    %% line searching
    while(true)
        %% gradient mapping
        d = v - s_crt.*grad;
        tmpd = d;
        %% proximal step (projection) for the squared group LASSO
        absd = abs(d(:));
        sabsd = sort(absd, 'descend');
        cumsortabsd = cumsum(sabsd);
        lineidx = [1:length(cumsortabsd)]';
        newlambda = lambda * s_crt;
        tmp = (newlambda./(1+newlambda.*lineidx)) .* cumsortabsd;
        tmp2 = sabsd - tmp;
        tmpidx = find(tmp2>0);
        inx_thresh = max(tmpidx);
        thresh = tmp(inx_thresh);
        d = tmpd;
        z = sign(d).*max(0, abs(d)-thresh*0.01);
        z = reshape(z, size(d));
        %% checking stop condition
        %% f value checking (inner_inner)
        z_mask = z.*supp_map;
        r = y-conv2(z_mask, k, 'same'); r = r.*selc_map_ext;
        f_crt = 0.5*(r(:)'*r(:));

        grad_mask = grad.*supp_map;
        x_prev_mask = x_est_prev.*supp_map;
        r2 = z_mask - x_prev_mask;
        f_hat = f_prev + grad_mask(:)'*r2(:) + (r2(:)'*r2(:))./(2*s_crt);
        if(f_crt<=f_hat)
            f = f_crt; f_prev = f;
            break;
        else
            s_crt = min(s_crt*beta, 0.01); % adjust the step size
        end
    end
    % line search -end-
    %% updating estimate value (x)
    x_est_prev = x_est;
    x_est = z;
    %     x_est = z.*supp_map;
    t_prev = t;
    t = (1+sqrt(1+4*t^2))/2;
    %% checking stop condition (for outer iterations)
    reg_val = 0;
    for i=1:length(group_indicator_cell)
        tmpC = group_indicator_cell{i};
        if(isempty(tmpC))
            break;
        end
        tmp = x_est.*tmpC;
        reg_val = reg_val + norm(tmp(:));
    end
    obj_crt = f + lambda*0.5*(reg_val^2);
    if((obj_crt-obj_prev)/(obj_crt-obj0)<stop_relative_rate)
        break;
    else
        obj_prev = obj_crt;
    end
end
fprintf('x_est_inner:%d\n', inner_iter);
x_out = x_est.*supp_map;
return
