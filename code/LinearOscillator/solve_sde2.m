function u_sol = solve_sde2(tspan, u0, eta, M, D, K, f, delta, C)
    % Generates a sample path from the linear oscillator problem 
    % using OU noise model.
    dt = tspan(2)-tspan(1);
    nt = length(tspan);
    d = length(u0)/3;
    u_sol = zeros(3*d,nt);
    u_sol(:,1) = u0(:);
    for i = 2:nt
        t_i = tspan(i);
        u_sol(:,i) = milstein_step2(t_i,u_sol(:,i-1), ...
            dt,eta,M,D,K,f,delta, C);
    end
end

%% Helper functions
function u_next = milstein_step2(t, u, dt, eta, M, D, K, f, delta, C)
    % take one time step of the Milstein scheme.
    n = length(u);
    assert(mod(n,3)==0);
    d = n/3;
    dWt = sqrt(dt)*randn(d,1);
    dZt = C*dWt;

    % evaluate forcing
    f_eval = f(t);
    
    % add drift
    u_next = zeros(3*d,1);
    
    Xt = u(1:d); Yt = u(d+1:2*d); Thetat = u((2*d+1):end);
    u_next(1:d) = Yt(:);
    u_next(d+1:2*d) = ((-M\(K*Xt(:)+D*Yt(:)+f_eval))+Thetat(:))*dt;

    % add diffusion to last term only
    u_next((2*d+1):end) = -delta*Thetat(:)+eta*dZt;
end