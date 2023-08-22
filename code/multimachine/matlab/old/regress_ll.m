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
    
%     warning('minimum bw for ksrlin manually tuned to 2*dx') 
    %%% may need to be changed for different case 
    bw_min = dx;
%     if state <= N
%         bw_min = 2*dx;   
%     elseif state <= 2*N
%         bw_min = 2*dx;
%     else
%         bw_min = 2*dx;
%     end
      
    % nb possible bandwidths
    % log spaced 
    bw = exp(linspace(log(bw_min),log(max(xx)-min(xx)),nb));
%       % linearly spaced + plug-in:
%       bw = sort([r0.h, linspace(dx,max(xx)-min(xx),nb-1)]); 

    %  k-fold CV for bandwidth selection
    [kscv_err, kscv_se] = ksrlin_cv(xx,yy,bw,nb,kf);

    % optimal bandwidth
    [~, ksmin_ind] = min(kscv_err); 
%     if ksmin_ind == nb
%         warning('CV minimized at maximal bw --> increase bandwidth range')
%     elseif ksmin_ind == 1
%         warning('CV minimized at minmal bw --> increase bandwidth range')
%     end
    
%     errors0 = [sqrt(ksmin),kscv_se(ksmin_ind)];
    
    % 1 St. Err. rule of thumb (increase regularity)
%     bw_1se = bw(find(bw(ksmin_ind:end) <= bw(ksmin_ind)+kscv_se(ksmin_ind),...
%                  1, 'last' ) - 1 + ksmin_ind);

    bw_1se = bw(ksmin_ind) + kscv_se(ksmin_ind);

    r_opt = ksrlin(xx,yy,bw_1se,xpts);   
    coeff0 = r_opt.f; 
    
end


%_______________________________________________________________________
function [cv_err, cv_se] = ksrlin_cv(xi,yj,bw,nb,kk)

    % kk-fold CV for local linear kernel smoothing regression
    nsize = numel(xi);
    cv_part = cvpartition(nsize, 'kfold',kk);
    ntest = cv_part.TestSize(1); 
    cv_err = zeros(1,nb); cv_se = cv_err;

    parfor ii = 1:nb

        % response data for current bw
%         r = ksrlin(xi,yj,bw(ii),xpts);

        % loop over folds
        cv_k = zeros(ntest,kk); 
        for jj = 1:kk
            
            test_ind = cv_part.test(jj);
            train_ind = cv_part.training(jj);

            % train/test 
            Y_te = ksrlin(xi(train_ind),yj(train_ind)...
                           ,bw(ii),xi(test_ind));

            cv_k(:,jj) = (yj(test_ind)-Y_te.f).^2;   
        end
        cv_err(ii) = mean(cv_k,'all');
        cv_se(ii) = sqrt(sum((cv_k - cv_err(ii)).^2,'all')/(ntest*(ntest-1)));
    end
    
end