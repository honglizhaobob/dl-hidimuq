function paths_mc = classical_mc(mc,dt,nt,u0,alpha,theta,C,H,D,P,wr,g,b)
% CLASSICAL_MC      Computes sample trajectories of classical multimachine 
% power model with Gaussian white noise via Euler's method.

% Inputs: 
%       n                       number of states
%
%
%       mc                      number of Monte Carlo trials
%
%
%       nt                      number to time steps to take
%
%
%       u0                      (mc x N) initial conditions
%
%
%       alpha, theta, C         parameters that define the noise
%                               dynamics. 

% Output:
%
%       paths_mc                (mc x N x nt) Monte Carlo trajectory
%                               data for all states. 

    % size of all states
    N = size(u0,2);
    n = round(N/4);

    % Allocate solution
    paths_mc = zeros(mc,4*n,nt);  
    for k = 1:mc 
        k
        % initialize at initial condition
        uk = reshape(u0(k,:),[],1);
        % integrate over nt time steps
        for j = 1:nt
            paths_mc(k,:,j) = uk;
            % take a step

            % add drift
            uk = uk + dt*rhs(uk,H,D,P,wr,g,b,theta);

            % add diffusion
            uk(3*n+1:end) = uk(3*n+1:end) + ...
                alpha*sqrt(2*theta)*C*(sqrt(dt).*randn(n,1));
         end
    end
end

function xP = rhs(x,H,D,P,wr,g,b,theta)
    % Right hand side of the stochastic multimachine model
    % noise is added separately during simulation. This rhs
    % only has drift.

    %   x(t) = [v(t), w(t), delta(t), eta(t)]
    [v,w,delta,eta]=split_vector(x);
    N = length(x);
    n = round(N/4);
    % right hand side vector
    dv = zeros(n,1);
    % matrix of pairwise angle differences
    tmp = delta(:)-delta(:)';
    dw = (wr/2)*(1./H).*( (-(w-wr).*D) + ...
        P - ( ( (g.*cos(tmp))+(b.*sin(tmp)) )*v ).*v + eta );
    ddelta = w-wr;
    deta = -theta*eta;
    % concatenate to (4nx1 vector)
    xP=[dv(:);dw(:);ddelta(:);deta(:)];
end