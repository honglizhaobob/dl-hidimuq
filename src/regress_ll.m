function coeff0 = regress_ll(xx,yy,xpts,nb,kf,dx)
    %  Local Linear (Gaussian) Kernel Regression with kf-fold bandwidth selection
    %  approximates f(x) in model Y = f(X) + e, where E[e] = 0
    
    % Input:
    %   xx := column vector of x data 
    %   yy := f(xx) + e
    %   xpts := query points to fit regression (column vector)
    %   nb>1 := total number of bandwidths for k-fold CV
    %   1<kf<=length(xx) := number of folds for CV 
    
    % Output:
    %   coeff0 := estimates regression function f(xpts) 

    % Optimal asymptotic bandwidth ==> assumes e ~ normal
%     r0 = ksrlin(xx,yy); 
    
    % nb possible bandwidths
      % log spaced 
      bw_min = dx; %%% WARNING! MANUALLY TUNED
      
      % nb possible bandwidths
      % log spaced 
      bw = exp(linspace(log(bw_min),log(max(xx)-min(xx)),nb));
%       % linearly spaced + plug-in:
      %bw = sort([r0.h, linspace(dx,max(xx)-min(xx),nb-1)]); 

    %  k-fold CV for bandwidth selection
    [kscv_err, kscv_se] = ksrlin_cv(xx,yy,bw,nb,kf);

    % optimal bandwidth
    [~, ksmin_ind] = min(kscv_err); 
    bw_1se = bw(ksmin_ind) + kscv_se(ksmin_ind);
    r_opt = ksrlin(xx,yy,bw_1se,xpts);   
    coeff0 = r_opt.f; 
end