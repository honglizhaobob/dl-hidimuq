%% Test 1: integrate oscillators with random initial conditions
clear; clc; rng("default");

d = 100;
u0 = randn(2*d,1);
tstart = 0.0;
tend = 10.0;
dt = 1e-3;
tspan = tstart:dt:tend;
% create mass, damping, stiffness matrices
M = eye(d);
K = diag(2.0*ones(1,d)) + diag(-1.0*ones(1,d-1),1) + diag(-1.0*ones(1,d-1),-1);
D = K;

% const forcing
f = @(t) zeros(d,1);
linear_oscillator = @(t, x) oscillator_rhs(t,x,M,D,K,f);

% generate trajectory
options = odeset("RelTol",1e-4);
[t, u_sol] = ode45(linear_oscillator, tspan, u0, options);