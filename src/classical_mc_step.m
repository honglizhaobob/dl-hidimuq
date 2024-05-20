function u1 = classical_mc_step(dt,u0,alpha,theta,C,H,D,P,wr,g,b)
    % For large systems, exceeds memory limit when allocating entire
    % arrays such as in `classical_mc.m`, therefore, we instead take 
    % step-by-step.

    % u0 should have shape (4*n x 1)
    
    % size of all states
    N = length(u0);
    n = round(N/4);
    uk = u0(:);
    % take a single step of dt
    uk = uk + dt*rhs(uk,H,D,P,wr,g,b,theta);
    % add diffusion
    uk(3*n+1:end) = uk(3*n+1:end) + ...
        alpha*sqrt(2*theta)*C*(sqrt(dt).*randn(n,1));
    u1 = uk(:);
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