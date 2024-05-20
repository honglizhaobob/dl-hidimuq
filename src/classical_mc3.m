function paths_mc = classical_mc3(mc,dt,nt,u0,alpha,theta,C,H,D,P,wr,g,b)
% CLASSICAL_MC3      Reimplementation of CLASSICAL_MC so that voltages 
% and noises are not stored. 
%
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
    paths_mc = zeros(mc,2*n,nt);  

    % get voltages
    volt = u0(:,1:n);
    for k = 1:mc 
        k
        % initialize at initial condition
        uk = reshape(u0(k,n+1:3*n),[],1);
        % OU noise process to be updated every loop, but not stored
        etak = reshape(u0(k,3*n+1:end),[],1);
        % voltage is constant throughout each path
        voltk = reshape(volt(k,:),[],1);
        % integrate over nt time steps
        for j = 1:nt
            paths_mc(k,:,j) = uk;

            % unpack
            w = uk(1:n);
            delta = uk(n+1:end);
            % compute angular differences 
            tmp = delta(:)-delta(:)';

            % add drift
            % modify angular velocity w
            uk(1:n) = w + dt*((wr/2)*(1./H).*( (-(w-wr).*D) + ...
                P - ( ( (g.*cos(tmp))+(b.*sin(tmp)) )*voltk ).*voltk + etak ));
            % modify angles delta
            uk(n+1:end) = delta + dt*(w-wr);

            % add diffusion to etak
            etak = etak - dt*theta*etak + alpha*sqrt(2*theta)*C*(sqrt(dt).*randn(n,1));
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