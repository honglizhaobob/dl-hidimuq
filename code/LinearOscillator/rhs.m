function dudt = rhs(t, u, M, D, K, f)
    % Right hand side of the linear oscillator, of dimension 2d
    assert(mod(length(u), 2)==0);
    d = length(u)/2;
    % unpack variables
    x = u(1:d);
    dxdt = u(d+1:end);
    % evaluate forcing 
    f_eval = f(t);
    % apply linear transformations
    dudt = [dxdt; -M\(D*dxdt-K*x+f_eval)];
end