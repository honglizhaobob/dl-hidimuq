function res = cond_exp(u, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper function to evaluate the conditional expectation
% term for the linear oscillator problem, given a full 
% state vector.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    u = u(:);
    N = length(u);
    % problem dimension
    d = round(N/3);
    % get X
    x = u(1:d);
    % get Y
    y = u(d+1:2*d);
    % inner product with stiffness matrix
    res = x'*params.K*y;
end