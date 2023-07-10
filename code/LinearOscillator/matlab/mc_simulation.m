%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main script for simulating stochastic linear oscillator energy
% observations, driven by OU noise for a few different parameter regimes.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc; rng("default");
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input MC simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% number of trials
nmc = 100;
% problem dimension
d = 100;
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

% energy function wrapper
potential_energy = @(x) energy(x(1:d), params.K);

% create SDE scheme
em_stepping.tgrid = tspan;
drift = @linear_oscillator_ou_noise_drift;
diffusion = @linear_oscillator_ou_noise_diffusion;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% preallocate
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
    % generate energy trajectory
    v_sol = sde_solve_energy(em_stepping, potential_energy);
    v_data(i,:) = v_sol(:);
end

%% Plot trajectories
figure(1); 
plot(tspan, squeeze(all_u_paths(:,:,1)), "LineWidth", 1.5);
figure(2);
plot(tspan, squeeze(all_u_paths(:,:,2)), "LineWidth", 1.5);

figure(1); title("Dim 1: MC = 100 Trajectories", "FontSize", 18);
figure(2); title("Dim 2: MC = 100 Trajectories", "FontSize", 18);

%% Save data after simulation
save_path = "../data/LinearOscillator/OU_noise_energy.mat";
save(save_path);

%% Load data when we start afresh
load_path = "../data/LinearOscillator/OU_noise_energy.mat";
load(load_path)

%% Simulate paths again to generate data for conditional expectation

% for linear oscillator problem, condition expectation is:
%   E[ X'*K*Y | v ]
cond_exp_data = zeros(nmc,nt);
for i = 1:nmc
    if mod(i,100)==0
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
    % generate energy trajectory
    cond_exp_path = sde_solve_func(em_stepping, @cond_exp, params);
    cond_exp_data(i,:) = cond_exp_path(:);
end

%% Save conditional expectation data
save_path = "../data/LinearOscillator/cond_exp.mat";
save(save_path);
%% Load conditional expectation data
load_path = "../data/LinearOscillator/cond_exp.mat";
load(load_path);
%% Run nonparameteric regression and visualize conditional expectation
figure(1);
% visualize X'*K*Y as a function of energy v(X)
num_v_gridpts = 10000;
v_grid = linspace(0,(1+0.2)*max(max(v_data)),num_v_gridpts);
for i = 1:nt
    figure(1);
    % fit spline
    fitted_curve = csaps(v_data(:,i),cond_exp_data(:,i),0.5,v_grid);
    scatter(v_data(:,i),cond_exp_data(:,i), ...
        "MarkerEdgeColor", "blue", ...
        "SizeData", 2);
    hold on;
    plot(v_grid, fitted_curve, "Color", "green");
    title(sprintf("$t = %0.2f$",i*dt), "Interpreter", "latex", ...
        "FontSize", 18, "LineWidth", 2);
    hold off;
    xlabel("Energy $V(X)$", "Interpreter", "latex", "FontSize", 18);
    ylabel("$E[X^TKY | V]$", "Interpreter", "latex", "FontSize", 18);
    xlim([-100 5000]);
    ylim([-20000 20000]);
    ax = gca;
    ax.XAxis.FontSize = 16;
    ax.YAxis.FontSize = 16;
    grid on;
end




%% Estimate density of potential
nx = 5e+3;
v_density = zeros(nx,nt);
xi = linspace(0, 5.5e+3, nx);
for i = 1:nt
    disp(i)
    % energy will never be negative
    [f, ~] = ksdensity(v_data(:,i),xi, ...
        "Support","Positive");
    v_density(:,i) = f;
    % normalize
    mass = trapz(xi,v_density(:,i));
    v_density(:,i) = v_density(:,i)./mass;
end
%% save density data
save_path = "../data/LinearOscillator/OU_noise_energy.mat";
save(save_path);
%%
for i = 1:nt
    figure(1);
    plot(xi, v_density(:,i), "LineWidth", 2.5, "Color", "red")
end






