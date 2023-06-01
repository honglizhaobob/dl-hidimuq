%% Test 1: integrate oscillators with random IC
clear; clc; rng("default");

d = 100;
u0 = randn(2*d,1);
tstart = 0.0;
tend = 10.0;
dt = 1e-2;
tspan = tstart:dt:tend;
% create mass, damping, stiffness matrices
M = eye(d);
D = eye(d);
K = randn(d,d);
K = (K'+K)/2;
% ensure positive definiteness
K = K+d*eye(d);

% const forcing
f = @(t) zeros(d,1);
% sine forcing with uniform freqs
%freqs = rand(d,1);
%f = @(t) sin(freqs*t);
linear_oscillator = @(t, x) oscillator_rhs(t,x,M,D,K,f);

% generate trajectory
options = odeset("RelTol",1e-4);
[t, u_sol] = ode45(linear_oscillator, tspan, u0, options);

%% Test 2: integrate oscillators with white noise and random IC
clear; clc; rng("default");

d = 100;
u0 = randn(2*d,1);
tstart = 0.0;
tend = 10.0;
dt = 1e-3;
tspan = tstart:dt:tend;
eta = 1;
% create mass, damping, stiffness matrices
M = eye(d);
D = eye(d);
K = randn(d,d);
K = (K'+K)/2;
% ensure positive definiteness
K = K+d*eye(d);

% const forcing
f = @(t) zeros(d,1);

% generate trajectory
u_sol = solve_sde(tspan, u0, eta, M, D, K, f);

%% Test 3: integrate oscillators with correlated noise and random IC
clear; clc; rng("default");

d = 10;
u0 = randn(3*d,1);
tstart = 0.0;
tend = 10.0;
dt = 1e-4;
tspan = tstart:dt:tend;
eta = 10.0;
% define parameters for OU process
delta = 0.01;
% uncorrelated noise
C = eye(d);

% correlated noise based on generator distance (lower triangular)
%C = exp_cov_kernel();
R = C*C';
% initial condition for OU noise
u0((2*d+1):end) = eta.*C*randn(d,1);

% create mass, damping, stiffness matrices
M = eye(d);
D = eye(d);
K = randn(d,d);
K = (K'+K)/2;
% ensure positive definiteness
K = K+d*eye(d);

% const forcing
f = @(t) zeros(d,1);

% generate trajectory
u_sol = solve_sde2(tspan, u0, eta, M, D, K, f, delta, C);
