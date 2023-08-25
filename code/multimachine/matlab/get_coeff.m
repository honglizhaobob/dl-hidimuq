function coeff0 = get_coeff(xx, yy, xpts_e, mode)
    % Computes advection coefficient for line energy marginal (on cell 
    % centers) based on regression from scatter data. 
    %
    % Model: yy = r(xx) + error, where E[error] = 0
    %
    % Exact solution is r(x0) = E[yy|xx = x0]
    if numel(xx)~=numel(yy)
        error("x and y are of different sizes. ");
    end
    xx = xx(:); yy = yy(:);
    % sort the data
    [xx,xx_ind] = sort(xx); yy = yy(xx_ind);
    % number of samples
    ns = length(xx);
    nx = length(xpts_e);
    dx = xpts_e(2)-xpts_e(1);
    if mode == "const"
        % constant regression (simply the average)
        coeff0 = ones(nx,1).*mean(yy);
    elseif mode=="lin"
        % linear regression (parameteric)
        oo = ones(ns,1); X = [oo,xx]; % include bias term
        beta = ((X'*X)\spdiags(oo,0,2,2))*X'*yy;
        coeff0 = beta(2)*xpts_e + beta(1);
    elseif mode == "llr"
        % local linear regression (nonparameteric)
        coeff0 = zeros(nx,1);
        % total number of bandwidths choices for CV: logarithmically spaced
        nb = 1; 
        % number of folds for k-fold CV
        kf = 10; 
        % Max Number of padding/extrapolation cells for learning coeff.
        pad = 2; 
        % Actual domain for regression (doing extrapolation at +/- pad*dx pts)
        j_ind = find(xpts_e>=(min(xx)-pad*dx) & xpts_e<=max(xx)+pad*dx);
        coeff0(j_ind) = regress_ll(xx,yy,xpts_e(j_ind),nb,kf,dx);

        % linearly extrapolate
        mx1 = (coeff0(j_ind(2))-coeff0(j_ind(1)))...
                 /(xpts_e(j_ind(2))-xpts_e(j_ind(1)));
         b1 = coeff0(j_ind(1)) - mx1*xpts_e(j_ind(1));
         coeff0(1:j_ind(1)-1) = mx1*xpts_e(1:j_ind(1)-1) + b1;
    
         mx2 = (coeff0(j_ind(end))-coeff0(j_ind(end-1)))...
                /(xpts_e(j_ind(end))-xpts_e(j_ind(end-1)));
         b2 = coeff0(j_ind(end)) - mx2*xpts_e(j_ind(end));
         coeff0(j_ind(end)+1:end) = mx2*xpts_e(j_ind(end)+1:end) + b2;

    elseif mode == "nn"
        % neural network prediction (considered nonparameteric)
        error("not implemented. ");
    else
        error("mode undefined.");
    end

    % check if any is nan and throw error
    if any(isnan(coeff0))
        error('coeff0 is NaN in get_coeff()')
    end
end