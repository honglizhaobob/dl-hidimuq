function dxdt = rhs2(x,H,D,P,wr,g,b)
    % Deterministic version (no OU noise) of the multimachine model
    % right hand side.

    %   x(t) = [v(t), w(t), delta(t)]
    n = round(length(x)/3);
    [v, w, delta] = split_vector2(x);
    % evaluate right hand side (no OU noise)
    % right hand side vector
    dxdt = zeros(3*n,1);
    % dvdt = 0
    % dwdt 
    % matrix of pairwise angle differences
    tmp = delta(:)-delta(:)';
    dxdt(n+1:2*n) = (wr/2)*(1./H).*( (-(w-wr).*D) + ...
        P - ( ( (g.*cos(tmp))+(b.*sin(tmp)) )*v ).*v );
    % d(delta)dt
    dxdt(2*n+1:end) = w-wr;
end

function [v,w,delta] = split_vector2(x)
    % local helper function to split the input `x` into 3 components, 
    % the ordering is assumed to be [v, w, delta]
    assert(mod(length(x),3)==0)
    n = round(length(x)/3);
    v = x(1:n);
    w = x(n+1:2*n);
    delta = x(2*n+1:3*n);
end