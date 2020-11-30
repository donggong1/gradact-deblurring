function k_out = solve_k_l2_conj(k_init, X, Y, opts)
  
% The problem that is
% being minimized is:
%
% min 1/2\|Xk - Y\|^2 + \lambda/2 \|k\|^2
% 
% Inputs:
% k_init = initial kernel, or scalar specifying size
% X = sharp image
% Y = blurry image  
% opts = options (see below)  
%
% Outputs:  
% k_out = output kernel
% 
% This function is adapted from the implementation for:
% Krishnan, Dilip, Terence Tay, and Rob Fergus. "Blind deconvolution using a normalized sparsity measure." CVPR 2011. IEEE, 2011.

  
% Defaults
if nargin == 3
  opts.lambda = 0;  
  % PCG parameters
  opts.pcg_tol = 1e-8;
  opts.pcg_its = 100;
  fprintf('Input options not defined - really no reg/constraints on the kernel?\n');
end

lambda = opts.lambda;
pcg_tol = opts.pcg_tol;
pcg_its = opts.pcg_its;

if (length(k_init(:)) == 1)
  k_init = zeros(k_init,k_init);
end
  
% assume square kernel
ks = size(k_init,1);
ks2 = floor(ks/2); 
  
% precompute RHS
for i = 1:length(X)
  flipX{i} = fliplr(flipud(X{i}));
  rhs{i} = conv2(flipX{i}, Y{i}, 'valid');
end

tmp = zeros(size(rhs{1}));
for i = 1 : length(X)
    tmp = tmp + rhs{i};
end
rhs = tmp;

k_out = k_init;

k_prev = k_out;
weights_l1 = lambda .* ones(size(k_prev)); %%% fixed values
k_out = local_cg(k_prev, X, flipX, ks,weights_l1, rhs, pcg_tol, pcg_its);
return
  

%% local implementation of CG to solve the reweighted least squares problem
function k = local_cg(k, X, flipX, ks, weights_l1, rhs, tol, max_its)
% is_test = logical(0);
Ak = pcg_kernel_core_irls_conv(k, X, flipX, ks,weights_l1);

r = rhs - Ak;

% r_value = zeros(1, max_its);
for iter = 1:max_its
  rho = (r(:)' * r(:));

  if (iter > 1)                      
    beta = rho / rho_1;
    p = r + beta*p;
  else
    p = r;
  end

  Ap = pcg_kernel_core_irls_conv(p, X, flipX, ks, weights_l1);

  q = Ap;
  alpha = rho / (p(:)' * q(:) );
  k = k + alpha * p; 
  r = r - alpha*q;                    
  rho_1 = rho;
  if (rho < tol)
    break;
  end
end
return


    
  
  
   
