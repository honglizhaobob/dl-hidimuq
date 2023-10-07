function [cv_err, cv_se] = ksrlin_cv(xi,yj,bw,nb,kk)
% KSRLIN_CV     local linear regression (LLR) with 
% parallelized k-fold cross validation. Divides data into k-partitions.
% For each iteration, hold out 1 partition and run
% LLR on data in the remaining (k-1) partitions 
% while computing mean squared error (MSE) on the holdout set. The bandwidth 
% selected in the end is the minimizer of MSE.
% 
%
% Inputs:
%
%   xi        : vector of independent variable observations.
%   yj        : vector of response variable observations.
%   bw        : vector of bandwidths to be selected from.
%   nb        : number of bandwidths.
%   kk        : number of partitions.
%
%
% Outputs:
%   cv_err    : MSE from cross validation (CV).
%   cv_se     : standard errors of CV.


    % k-fold CV for local linear kernel smoothing regression
    nsize = numel(xi);
    cv_part = cvpartition(nsize, 'kfold',kk);
    ntest = cv_part.TestSize(1); 
    cv_err = zeros(1,nb); cv_se = cv_err;

    parfor ii = 1:nb
        % loop over folds
        cv_k = zeros(ntest,kk); 
        for jj = 1:kk
            
            test_ind = cv_part.test(jj);
            train_ind = cv_part.training(jj);

            % train/test 
            Y_te = ksrlin(xi(train_ind), ...
                yj(train_ind),bw(ii),xi(test_ind));

            cv_k(:,jj) = (yj(test_ind)-Y_te.f).^2;   
        end
        cv_err(ii) = mean(cv_k,'all');
        cv_se(ii) = sqrt(sum((cv_k - cv_err(ii)).^2,'all')/(ntest*(ntest-1)));
    end
    
end

