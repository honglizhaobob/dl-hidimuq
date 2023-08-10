function [v, w, delta, eta] = split_vector(x)
    % helper function, splits vector x(t) into 4 components
    N = length(x);
    assert(mod(N,4)==0);
    n = round(N/4);
    % split vector
    v = x(1:n);
    w = x(n+1:2*n);
    delta = x(2*n+1:3*n);
    eta = x(3*n+1:end);
end