%% Test 1: integrate oscillators with random IC
clear; clc; rng("default");

d = 20;
u0 = randn(2*d,1);
tstart = 0.0;
tend = 10.0;
dt = 1e-3;
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

d = 20;
u0 = randn(2*d,1);
tstart = 0.0;
tend = 10.0;
dt = 1e-3;
tspan = tstart:dt:tend;
eta = 1.0;
% create mass, damping, stiffness matrices
M = eye(d);
D = eye(d);
K = randn(d,d);
K = (K'+K)/2;
% ensure positive definiteness
K = K+d*eye(d);
% const forcing
f = @(t) zeros(d,1);
% wrap params
params.M = M;
params.D = D;
params.K = K;
params.f = f;

% create SDE scheme
em_stepping.u0 = u0;
em_stepping.tgrid = tspan;
drift = @linear_oscillator_white_noise_drift;
diffusion = @linear_oscillator_white_noise_diffusion;
em_stepping.step = @(t, u) euler_maruyama_step(t, u, drift, diffusion, dt, eta, params);
% generate trajectory
u_sol = sde_solve(em_stepping);
plot(u_sol(1:d, :)');

%% Test 3: integrate oscillators with correlated noise and random IC
clear; clc; rng("default");

d = 20;
u0 = randn(3*d,1);
tstart = 0.0;
tend = 20.0;
dt = 1e-3;
tspan = tstart:dt:tend;
eta = 1.0;
% define parameters for OU process
delta = 1e-2;


% create mass, damping, stiffness matrices
M = eye(d);
D = eye(d);
K = randn(d,d);
K = (K'+K)/2;
% ensure positive definiteness
K = K+d*eye(d);

% correlated noise based on initial distance
abs_dist = abs(u0(1:d)-u0(1:d)');
lambda = 1;
R = exp_cov(abs_dist,lambda);
C = chol(R)';
% initial condition for OU noise
u0((2*d+1):end) = eta.*C*randn(d,1);

% const forcing
f = @(t) zeros(d,1);

% wrap parameters
params.M = M;
params.D = D;
params.K = K;
params.f = f;
params.eta = eta;
params.delta = delta;
params.C = C;

% create SDE scheme
em_stepping.u0 = u0;
em_stepping.tgrid = tspan;
drift = @linear_oscillator_ou_noise_drift;
diffusion = @linear_oscillator_ou_noise_diffusion;
em_stepping.step = @(t, u) euler_maruyama_step(t, u, drift, diffusion, dt, eta, params);

% generate trajectory
u_sol = sde_solve(em_stepping);
figure(1);
plot(u_sol(1:d, :)');
figure(2);
plot(u_sol((2*d+1):end, :)')