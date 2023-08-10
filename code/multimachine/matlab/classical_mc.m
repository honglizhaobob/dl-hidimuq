function paths_mc = classical_mc(mc,dt,nt,u0,alpha,theta,C,H,D,P,wr,g,b)
    % Computes sample trajectories of classical multimachine power model 
    %   with Gaussian white noise via Euler's method

    % Inputs: 
    %       n                       number of states
    %
    %
    %       mc                      number of Monte Carlo trials\
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