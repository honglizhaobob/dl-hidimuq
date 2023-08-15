function r = ksrlin(x,y,h,xpts)
% KSRLIN   Local linear (Gaussian) kernel smoothing regression (LLR)
%
% Input: (variable arguments, min of 2, max of 4)
%   x:=    (required) real vector of predictor data
%   y:=    (required) response data --> same size as x
%   h:=    (optional) bandwidth for Gaussian kernel
%   xpts:= (optional) query points 
%   kern:= (optional) kernel --> Gaussian default, but can select "tricube" 
%
% Output: (Variable, min of 0, max of 1)
%   r = ksrlin(x,y) returns the LLR in structure r, where
%       r.f (col. vect.) is the regression function defined on (col. vect.) 
%       of query nodes r.x. 
%       Default: 
%          r.x (xpts) = linspace(min(x),max(x),100)', 
%          r.h (bandwidth h) = optimal asymptotic plug-in for Gaussian kernel
%       Also returns r.n = length(x)
%       If ksrlin(x,y) is without assigning output, then the data scatter 
%       and regression function are plotted.
%
%   r = ksrlin(x,y,h)--> same, but with specified bandwidth r.h = h
%
%   r = ksrlin(x,y,h,xpts)--> same, but with specified bandwidth r.h = h
%       and (col. vect.) query nodes r.x = xpts.
%
% Algorithm:
%   The kernel regression is a non-parametric approach to
%   estimate the regression function (conditional expectation) 
%   f(X) = E[Y|X] in the model Y = f(X) + e, where E[e]=0.
%
%   Normal (NW) kernel regression is a local constant estimator. 
%   The extension of local linear estimator is obtained by solving the 
%   least squares problem:
%
%   min sum (y-alpha-beta(x-X))^2 kerf((x-X)/h).
%
%   The local linear estimator can be given an explicit formula:
%
%   f(x0) = 1/n sum((s2-s1*(x0-x)).*kerf((x0-x)/h).*y)/(s2*s0-s1^2),
%
%   where si = sum((x0-x)^i*kerf((x0-x)/h))/n. Compared with local constant
%   estimator, the local linear estimator can improve the estimation near the
%   edge of the region over which the data have been collected. It corrects
%   boundary bias exactly to 1st-order, which is crucial for extrapolation.
%
%   In interior regions of high curvature, the linear estimator is known to
%   bias the results. Using higher-order polynomial regression can improve 
%   the linear bias in these regions, but at the price of increased 
%   variance and more Computational costs. It also does not improve upon
%   the linear estimators boundary estimates, but still increases variance
%
%   Note: if the regression function is inherently flat, the local constant
%   (NW) estimator may out perform the local linear estimator.
%
%   See also gkde, ksdensity, ksr
%   Reference:
%   Bowman, A.W. and Azzalini, A., Applied Smoothing Techniques for Data
%   Analysis, Clarendon Press, 1997 (p.50) 
% 
%  Example: smooth curve with noise
    %{
    x = 1:100;
    y = sin(x/10)+(x/50).^2;
    yn = y + 0.2*randn(1,100);
    r = ksrlin(x,yn);
    r1 = ksr(x,yn); % downloadable from FEX ID 19195
    plot(x,y,'b-',x,yn,'co',r.x,r.f,'r--',r1.x,r1.f,'m-.','linewidth',2)
    legend('true','data','local linear','local constant','location','northwest');
    title('Gaussian kernel regression')
    %}

% Originally by Yi Cao at Cranfield University on 12 April 2008.
% Last edited by Tyler Maltba, Argonne Nat. Lab., UC Berkeley, 14 June 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check input and output
narginchk(2,4);
nargoutchk(0,1);

% % I do this outside of the function:
% if numel(x) ~= numel(y)
%     error('x and y are in different sizes.');
% end
% x = x(:);
% y = y(:);
%
% % clean missing or invalid data points
% inv = (x~=x)|(y~=y)|(abs(x)==Inf)|(abs(y)==Inf) ;
% x(inv)=[];
% y(inv)=[];

% Default parameters and method
a = min(x); b = max(x);
if nargin < 4
    N = 100;
    xpts = linspace(a,b,N)';
elseif ~isvector(xpts) || ~isreal(xpts)
    error('xpts must be a real vector')
else
    xpts = xpts(:);
    N = length(xpts);
end
r.x = xpts;
r.n = length(x);

if nargin<3
    % optimal bandwidth suggested by Bowman and Azzalini (1997) p.31
    hx = median(abs(x-median(x)))/0.6745*(4/3/r.n)^0.2;
    hy = median(abs(y-median(y)))/0.6745*(4/3/r.n)^0.2;
    h = sqrt(hy*hx);
elseif ~isscalar(h) || h<0
    error('h must be a scalar > 0')
end
r.h = h;
r.f = zeros(N,1);

% Gaussian kernel function (Default)
kerf = @(z)exp(-0.5*z.^2)/sqrt(2*pi);

% Local Linear Regression
for k = 1:N
    
%     B = [ones(r.n,1),x];
%     W = spdiags(kerf((xpts(k)-x)/h),1,r.n,r.n);
%     r.f(k) = [1,xpts(k)]*((B'*W*B)\speye(2))*B'*W*y;
 
    % instead of above, directly calculate all of the matrix products --> faster
    d = xpts(k)-x;  z = kerf(d/h);
    s1 = d.*z;  s2 = sum(d.*s1);  s1 = sum(s1);
    r.f(k) = sum((s2-s1*d).*z.*y)/(s2*sum(z)-s1^2);    
    % where si = sum((x-X)^i*kerf((x-X)/h))/n, and 
    % f(x) = 1/n sum((s2-s1*(x-X)).*kerf((x-X)/h).*Y)/(s2*s0-s1^2)
end

% Plot
if ~nargout
    plot(x,y,'ok','markersize',2)
    hold on
    plot(r.x,r.f,'b','linewidth',2)
    ylabel('f(x)')
    xlabel('x')
    title('Loc. Lin. Kernel Smoothing Regression');
end