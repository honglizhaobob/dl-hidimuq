function u_sol = solve_sde(tspan, u0, eta, M, D, K, f)
    % Generates a sample path from the stochastically driven
    % coupled linear oscillators problem. The solution assumes
    % Gaussian white noise with amplitude eta.
    dt = tspan(2)-tspan(1);
    nt = length(tspan);
    d = length(u0)/2;
    u_sol = zeros(2*d,nt);
    u_sol(:,1) = u0(:);
    for i = 2:nt
        t_i = tspan(i);
        u_sol(:,i) = milstein_step(t_i,u_sol(:,i-1), ...
            dt,eta,M,D,K,f);
    end
end

%% Helper functions
function u_next = milstein_step(t, u, dt, eta, M, D, K, f)
    % take one time step of the Milstein scheme.
    n = length(u);
    assert(mod(n,2)==0)
    dWt = sqrt(dt)*randn(n/2,1);

    % add drift
    u_next = u+oscillator_rhs(t, u, M, D, K, f)*dt; 

    % add diffusion
    u_next(n/2+1:end) = u_next(n/2+1:end)+eta*dWt;
end