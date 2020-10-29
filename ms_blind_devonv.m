function [kernel, x_est_latent, time_k_estimation] ...
    = ms_blind_devonv(blur_img, x_mpl_opts, k_est_opts, opts)

%% prepareration
if(size(blur_img,3)==3)
    y = rgb2gray(blur_img);
else
    y = blur_img;
end
if opts.gamma_correct~=1
    y = y.^opts.gamma_correct;
end
% crop part image for kernel estimation
if (~isempty(opts.kernel_est_win))
    tmp_win = opts.kernel_est_win;
    y = y(tmp_win(1):tmp_win(2), tmp_win(3):tmp_win(4));
end

% image size
y_h = size(y,1);
y_w = size(y,2);

% kernel size
ks = opts.kernel_size;
khs = floor(ks/2);

% feature filters (derivative filters)
fd{1} = [-1 1; 0 0];
fd{2} = [-1 0; 1 0]; % x,y filters
num_fd = length(fd);

% l2 norm of gradient images, if required
l2norm = 6;

% set kernel size for coarsest level
ret = sqrt(0.5);
%%
maxitr=max(floor(log(4/min(opts.kernel_size))/log(ret)),0);
num_scales = maxitr + 1;
fprintf('Number of scales: %d\n', num_scales);
retv=ret.^[0:maxitr];
ksize_list=ceil(bsxfun(@times, opts.kernel_size, retv(:)));
ksize_list = ksize_list';
%ksize_list=ceil(opts.kernel_size.*retv);
ksize_list=ksize_list+(mod(ksize_list,2)==0);

tic;
%% multi-scale
for s = num_scales:-1:1
    %% init/resize kernl
    if (s == num_scales)
        % at coarsest level, initialize kernel
        k_scl = init_kernel(ksize_list(:,s));
        [ksizeh_scl, ksizew_scl] = size(k_scl);
        tmpkh = ksizeh_scl; tmpkw = ksizew_scl;
    else
        % upsample kernel from previous level to next finer level
        ksizeh_scl = ksize_list(1,s);
        %         ksizew_scl = ksize_list(2,s) % always square kernel assumed
        % resize kernel from previous level
        if(numel(opts.kernel_size)==1)
            k_scl = resizeKer(k_scl,1/ret,ksize_list(s),ksize_list(s));
            tmpkh = ksizeh_scl; tmpkw = tmpkh;
        else
            k_scl = resizeKer(k_scl,1/ret,ksize_list(1,s),ksize_list(2,s));
            tmpkh = ksize_list(1,s); tmpkw = ksize_list(2,s);
        end
    end
    %% resize blur image according to the ratio
    y_scl=downSmpImC(y,retv(s));
    fprintf('\n-Processing scale %d/%d; kernel size %dx%d; image size %dx%d\n', ...
        s, num_scales, tmpkh, tmpkw, size(y_scl,1), size(y_scl,2));
    
    %%
    for i = 1:num_fd
        grad_y{i} = conv2(y_scl, fd{i}, 'valid');
        tmphs(i) = size(grad_y{i}, 1); tmpws(i) = size(grad_y{i}, 2);
    end
    %%
    for i = 1:num_fd
        tmp = grad_y{i};
        tmp = tmp*l2norm/norm(tmp(:));
        grad_y{i} = tmp;
    end
    clear tmp;
    % -----------
    if(s==1)
        %% x_mpl setting
        x_mpl_opts.lambda_x = x_mpl_opts.lambda_x_fine;
    end
    
    if(s==1)
        opts.is_k_lambda_dec = logical(1);
    else
        opts.is_k_lambda_dec = logical(0);
    end
    
%     tic;
    x_scl = grad_y;
    [x_scl, k_scl] = ss_blind_deconv(grad_y, x_scl, k_scl, ...
        x_mpl_opts, k_est_opts, opts);
%     toc;
    [x_scl, k_scl] = center_kernel_separate(x_scl, k_scl);
    if (s == 1)
        kernel = k_scl;
        if(opts.k_final_iso_prun ...
           && (strcmp(k_est_opts.type, 'l2_conj')||strcmp(k_est_opts.type, 'irls')))
            CC = bwconncomp(kernel,8);
            for ii=1:CC.NumObjects
                currsum=sum(kernel(CC.PixelIdxList{ii}));
                if currsum<0.1
                    kernel(CC.PixelIdxList{ii}) = 0;
                end
            end
        end
        if(opts.k_thresh>0)
            kernel(kernel(:) < max(kernel(:))/opts.k_thresh) = 0;
        else
            kernel(kernel(:)<0) = 0;
        end
        if(opts.is_kernel_diff)
            kernel = conv2(kernel, fspecial('gaussian',3,0.4), 'same');
        end
        kernel = kernel / sum(kernel(:));
    end
end
toc;
time_k_estimation = toc;
x_est_latent = x_scl;
return


%% initialize kernel
function [k] = init_kernel(minsize)
if(numel(minsize)==1)
    h = minsize;
    w = h;
elseif(numel(minsize)==2)
    h = minsize(1);
    w = minsize(2);
else
    k=nan;
    fprintf('Error size of input variable.\n');
    return;
end
k = zeros(h, w);
% k((h+1)/2,:)=1;
% k = k./sum(k(:));
k((h - 1)/2, (w - 1)/2:(w - 1)/2+1) = 1/2;


% k((h - 1)/2+1, (w - 1)/2+1) = 1/2;
% k((minsize + 1)/2, (minsize - 1)/2:(minsize - 1)/2+1) = 1/2;
%     k((minsize + 1)/2-1, (minsize + 1)/2:(minsize + 1)/2+1) = 1/2;
% k = imresize(gt_k, [minsize, minsize]);
return


%%
function k=resizeKer(k,ret,k1,k2)
% levin's code
k=imresize(k,ret);
k=max(k,0);
k=fixsize(k,k1,k2);
if(max(k(:))>0)
    k=k/sum(k(:));
else
    fprintf('ERROR on Kernel.\n');
end
return

%% refer to Levin's code
function sI=downSmpImC(I,ret)
if (ret==1)
    sI=I;
    return;
end
sig=1/pi*ret;

g0=[-50:50]*2*pi;
sf=exp(-0.5*g0.^2*sig^2);
sf=sf/sum(sf);
csf=cumsum(sf);
csf=min(csf,csf(end:-1:1));
ii=find(csf>0.05);
sf=sf(ii);
sum(sf);
I=conv2(sf,sf',I,'valid');

[gx,gy]=meshgrid([1:1/ret:size(I,2)],[1:1/ret:size(I,1)]);

sI=interp2(I,gx,gy,'bilinear');
return

function nf=fixsize(f,nk1,nk2)
[k1,k2]=size(f);

while((k1~=nk1)|(k2~=nk2))
    
    if (k1>nk1)
        s=sum(f,2);
        if (s(1)<s(end))
            f=f(2:end,:);
        else
            f=f(1:end-1,:);
        end
    end
    
    if (k1<nk1)
        s=sum(f,2);
        if (s(1)<s(end))
            tf=zeros(k1+1,size(f,2));
            tf(1:k1,:)=f;
            f=tf;
        else
            tf=zeros(k1+1,size(f,2));
            tf(2:k1+1,:)=f;
            f=tf;
        end
    end
    
    if (k2>nk2)
        s=sum(f,1);
        if (s(1)<s(end))
            f=f(:,2:end);
        else
            f=f(:,1:end-1);
        end
    end
    if (k2<nk2)
        s=sum(f,1);
        if (s(1)<s(end))
            tf=zeros(size(f,1),k2+1);
            tf(:,1:k2)=f;
            f=tf;
        else
            tf=zeros(size(f,1),k2+1);
            tf(:,2:k2+1)=f;
            f=tf;
        end
    end
    [k1,k2]=size(f);
end
nf=f;
return