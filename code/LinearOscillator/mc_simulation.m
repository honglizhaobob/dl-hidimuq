%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main script for simulating stochastic linear oscillator energy
% observations, driven by OU noise for a few different parameter regimes.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; rng("default");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input MC simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% number of trials
nmc = 1e+3;
% problem dimension
d = 50;
% time grid
tstart = 0.0;
tend = 10.0;
dt = 2e-3;
tspan = tstart:dt:tend;
nt = length(tspan);
% noise amplitude
eta = 2.0;
% speed of mean-reversion
delta = 1e-2;

% create mass, damping, stiffness matrices
M = eye(d);
D = eye(d);
K = randn(d,d);
K = (K'+K)/2;
% ensure positive definiteness
K = K+d*eye(d);

% const forcing
f = @(t) zeros(d,1);

% wrap parameters
params.M = M;
params.D = D;
params.K = K;
params.f = f;
params.eta = eta;
params.delta = delta;

% create SDE scheme
em_stepping.tgrid = tspan;
drift = @linear_oscillator_ou_noise_drift;
diffusion = @linear_oscillator_ou_noise_diffusion;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% preallocate
u_data = zeros(nmc,3*d,nt);
v_data = zeros(nmc,nt);
for i = 1:nmc
    if mod(i,10)==0
        fprintf("> MC simulation = %d\n", i);
    end
    % random initial condition
    u0 = randn(3*d,1);
    % correlated noise based on initial distance
    abs_dist = abs(u0(1:d)-u0(1:d)');
    lambda = 1;
    R = exp_cov(abs_dist,lambda);
    C = chol(R)';
    % initial condition for OU noise
    u0((2*d+1):end) = eta.*C*randn(d,1);

    % update SDE parameters
    params.C = C;
    em_stepping.step = @(t, u) euler_maruyama_step(t, u, drift, diffusion, dt, eta, params);
    em_stepping.u0 = u0;
    % generate trajectory
    u_sol = sde_solve(em_stepping);
    % store
    u_data(i, :, :) = u_sol;
    % compute energy
    for j = 1:nt
        v_data(i,j) = energy(u_sol(1:d,j),params.K); 
    end
end

%% Save data
save_path = "./data/LinearOscillator/OU_noise_energy.mat";
save(save_path);
%% Estimate density of potential
nx = 1e+3;
v_density = zeros(nx,nt);
xi = linspace(0, 5e+3, nx);
for i = 1:nt
    disp(i)
    [f, ~] = ksdensity(v_data(:,i),xi);
    v_density(:,i) = f;
end
%%
for i = 1:nt
    figure(1);
    plot(xi, v_density(:,i))
    ylim([0 0.015])
end
