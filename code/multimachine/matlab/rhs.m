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