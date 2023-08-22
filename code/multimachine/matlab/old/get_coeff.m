function [coeff, coemax] = get_coeff(nn,method,mc,N,osc,state,u,xi,nxi,v,hh,dd,pm,wwr,b,g,xpts_e,dx)
 
   
    % Model: yy = r(xx) + e, where E[e] = 0 
    % Under MSE loss, exact solution is r(x0) = E[yy|xx = x0]
    % For us, yy is part of the advection coeff. that can't be pulled out
    %   of the conditional expectation.
    
    xx = u(state,:)';
    
    if state <= N   % Speed/velocity
  
        dmat = repmat(u(state+N,:),N,1) - u(N+1:2*N,:);
        yy = (pm - v(osc,:).*sum(v.*(repmat(g(osc,:)',1,mc).*cos(dmat)...
            + repmat(b(osc,:)',1,mc).*sin(dmat))) + xi(state,:))';
        
    elseif state <= 2*N  % Angle/phase
        
        yy = u(state-N,:)';
        if nxi == 2
            yy = yy + xi(state,:)';
        end
        yy = yy - wwr;

    else
        error('Not appropriate state')
    end
    
    % clean missing or invalid data points, then sort
    if numel(xx) ~= numel(yy)
        error('x and y are in different sizes.');
    end
    xx = xx(:);  yy = yy(:);
    inv = (xx~=xx)|(yy~=yy)|(abs(xx)==Inf)|(abs(yy)==Inf);
    xx(inv)=[];
    yy(inv)=[];
    [xx,xx_ind] = sort(xx); yy = yy(xx_ind);
    
    if  ~isreal(xx) || ~isreal(yy)
        error('Need xx and yy real col. vector')
    end
    
    ns = length(xx);       % number of samples
    nx = length(xpts_e);   % number of query points
    
%     if nn<340 
    
    % Allocate coeff
    coeff0 = zeros(nx,1); 
    
    if method == "lin"
        oo = ones(ns,1); X = [oo,xx];
        beta = ((X'*X)\spdiags(oo,0,2,2))*X'*yy;
        coeff0 = beta(2)*xpts_e + beta(1);

%         ci = .95;  % confidence level
%         mdl1 = fitlm(xx,yy); %'RobustOpts','on'
%         if coefTest(mdl1) >= 1-ci  % Test p-value of F-stat
%             coeff0 = ones(nx,1)*mean(yy);
% %             slope = 0;
%         else
%             coeff0 = predict(mdl1,xpts_e);
% %             slope = mdl1.Coefficients.Estimate(2);
%         end

    else 
        % method == 'llr'
        % total number of bandwidths choices for CV: logarithmically spaced
        nb = 16; 
        % number of folds for k-fold CV
        kf = 5; 
        % Max Number of padding/extrapolation cells for learning coeff.
        pad = ceil(3*std(xx)/dx); 
        
        % Actual domain for regression (doing extrapolation at +/- pad*dx pts)
        j_ind = find(xpts_e>=(min(xx)-pad*dx) & xpts_e<=max(xx)+pad*dx);
        
        % check inputs for regress_ll function
        if mod(length(xx),kf)~=0 % needs to divide number of samples
            error("Number of CV folds doesn't divide sample size")
        end 
        if ~(nb>1 && rem(nb,1)==0 && isscalar(nb) && isreal(nb))
            error('Need (real) integer nb > 1')
        end
        if ~(kf>1 && kf<=length(xx) && rem(kf,1)==0 && isscalar(kf) && isreal(kf))
            error('Need (real) integer 2 <= kf <= length(xx)')
        end
        coeff0(j_ind) = regress_ll(xx,yy,xpts_e(j_ind),nb,kf,dx);
        
%         % linear extrapolation
%         mx1 = (coeff0(j_ind(2))-coeff0(j_ind(1)))...
%                 /(xpts_e(j_ind(2))-xpts_e(j_ind(1)));
%         b1 = coeff0(j_ind(1)) - mx1*xpts_e(j_ind(1));
%         coeff0(1:j_ind(1)-1) = mx1*xpts_e(1:j_ind(1)-1) + b1;
% 
%         mx2 = (coeff0(j_ind(end))-coeff0(j_ind(end-1)))...
%                /(xpts_e(j_ind(end))-xpts_e(j_ind(end-1)));
%         b2 = coeff0(j_ind(end)) - mx2*xpts_e(j_ind(end));
%         coeff0(j_ind(end)+1:end) = mx2*xpts_e(j_ind(end)+1:end) + b2;

%         % linear extrapolation
%         mx1 = (coeff0(j_ind(2))-coeff0(j_ind(1)))...
%                 /(xpts_e(j_ind(2))-xpts_e(j_ind(1)));
%         b1 = coeff0(j_ind(1)) - mx1*xpts_e(j_ind(1));
%         coeff0(j_ind(1)-10:j_ind(1)-1) = mx1*xpts_e(j_ind(1)-10:j_ind(1)-1) + b1;
% 
%         mx2 = (coeff0(j_ind(end))-coeff0(j_ind(end-1)))...
%                /(xpts_e(j_ind(end))-xpts_e(j_ind(end-1)));
%         b2 = coeff0(j_ind(end)) - mx2*xpts_e(j_ind(end));
%         coeff0(j_ind(end)+1:j_ind(end)+10) = mx2*xpts_e(j_ind(end)+1:j_ind(end)+10) + b2;
% 
%         coeff0(j_ind(1)-10:j_ind(1)-1) = 0; 
%         coeff0(j_ind(end)+1:j_ind(end)+10) = 0;
%         coeff0 = smoothdata(coeff0,'gaussian',7);

        coeff0(1:j_ind(1)-1) = 0; 
        coeff0(j_ind(end)+1:end) = 0;
        coeff0 = smoothdata(coeff0,'gaussian',9);
    end
    
    if state <= N % Speed/velocity
        
        coeff = 0.5*wwr*(dd*(wwr- xpts_e) + coeff0)/hh; 
        
    elseif state <= 2*N % Angle/phase
        
        coeff = coeff0;
    end
    
    % check is nan
    if max(isnan(coeff))==1
        error('coeff is NaN in get_coeff()')
    else
        coemax = max(abs(coeff));
        if coemax == Inf
            error('coeff in fget_coeff() blows up')
        end
    end
    
%     else
%         coeff = zeros(nx,1);
%         coemax = 0;
%     end
    
%   if mod(nn,40)==0 
%       disp(nn)
%     figure(4)
%     plot(xx,yy,'ko','markersize',2); 
%     hold on
%     plot(xpts_e,coeff0,'-b','linewidth',2); 
%     set(gca,'linewidth',1.5, 'fontsize',20)
%     title('Coeff0 Estimate'); xlabel('X'); 
% %     xlim([min(xx,[],'all') max(xx,[],'all')]);
%     hold off
%     drawnow
%     aa=1;
%   end
    
   