function u1 = classical_mc_step(dt,u0,alpha,theta,C,H,D,P,wr,g,b)
    % For large systems, exceeds memory limit when allocating entire
    % arrays such as in `classical_mc.m`, therefore, we instead take 
    % step-by-step.

    % u0 should have shape (4*n x 1)
    
    % size of all states
    N = size(u0,2);
    n = round(N/4);
    uk = u0(:);
    % take a single step of dt
    uk = uk + dt*rhs(uk,H,D,P,wr,g,b,theta);
    % add diffusion
    uk(3*n+1:end) = uk(3*n+1:end) + ...
        alpha*sqrt(2*theta)*C*(sqrt(dt).*randn(n,1));
    u1 = uk(:);
end