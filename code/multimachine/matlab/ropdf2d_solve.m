%     Hongli Zhao, Givens Associate, Argonne Nat. Lab.,
%     CAM PhD Student, UChicago
%     Last updated: Aug 15, 2023

% Solving 2d joint RO-PDF equation of energies of two lines
% Define simulation parameters
%%
clear; rng('default');

% Make sure Matpower files added to path 
run("./matpower7.1/startup.m");

% choose case from matpower7.1/data
[Pm, g, b, vi, d0, success] = runpf_edited('case9.m');

if success ~= 1
    error('runpf_edited to not run successfully')
end

% MatPower outputs values for the system in equilibrium:
% Pm = vector of bus power injections
% g = admittance matrix for cosine (generators)
% b = admittance matrix for sine (buses)
% vi = voltage magnitudes
% d0 = initial angles delta(0)

n = length(d0);       % Number of buses
H = ones(n,1);        % Inertia coefficients
D = ones(n,1);        % Damping coefficients
wr = 1;               % base speed

% Governor constants 
R = 0.02;              % Droop
T1 = 0;                % Transient gain time
T2 = 0.1;              % Governor time constant

mc = 5000;             % Number of MC paths
tf = 50.0;               % final time for simulation
dt = 0.01;             % learn PDE coefficients in increments of dt
time = 0:dt:tf;        % coarse uniform time grid
tt = time;
nt = length(time);     % number of time steps

sig = 0.1*ones(n,1);   % SDE noise amplitudes for speeds
dt0 = 1e-2;            % time step for SDE --> must divide dt

if mod(dt,dt0)~=0
    error('dt0 does not divide dt')
end

%% Define random initial conditions
% Allocate random initial conditions:
u0 = zeros(mc,4*n);

% Random Initial speeds (Gaussian)
mu_w = 1; sd_w = 0.1;
u0_w = sd_w*randn(mc,n) + mu_w;

% Random Initial angles Gaussian
sd_d = 10.0 * pi/180.0; % sd_d = 10.0 degrees, mean around optimal d0
u0_d = sd_d*randn(mc,n) + reshape(d0,1,[]);

% Random voltages . (folded gaussian, mean at optimal vi)
sd_v = mean(vi)*0.01;
v = abs(sd_v*randn(mc,n) + reshape(vi,1,[]));

% Random initial conditions for OU noise
theta = 1.0;                % drift parameter
alpha = 0.05;               % diffusion parameter

% define covariance matrix
case_number = 9;
mode = "const";
reactance_mat = [];
susceptance_mat = [];
R = cov_model(case_number, mode, reactance_mat, susceptance_mat);
C = chol(R)';
eta0 = mvnrnd(zeros(n,1),(alpha^2)*R,mc);

% store in initial condition, ordering [v; w; delta; eta]
u0(:, 1:n) = v;      
u0(:, n+1:2*n) = u0_w;
u0(:, 2*n+1:3*n) = u0_d;
u0(:, 3*n+1:end) = eta0;


%% Simulate Monte Carlo trajectories
tic
%(3*N x nt x mc)
paths_mc = classical_mc(mc,dt,nt,u0,alpha,theta,C,H,D,Pm,wr,g,b);
toc

%% Visualize Monte Carlo data
close all
tt = time';
% All mc trials
f = figure(1);
f.Position = [500 500 1240 400];
subplot(1,4,1)
plot(tt,squeeze(paths_mc(1,1:n,:)), "LineWidth", 1.2); 
title('MC voltages for i = 1')
set(gca,'linewidth',1.5, 'fontsize',20); xlabel('t')
xlim([0 tt(end)])

subplot(1,4,2)
plot(tt,squeeze(paths_mc(1,n+1:2*n,:)), "LineWidth", 1.2); 
title('MC speeds for i = 1')
set(gca,'linewidth',1.5, 'fontsize',20); xlabel('t')
xlim([0 tt(end)])

subplot(1,4,3)
plot(tt,squeeze(paths_mc(1,2*n+1:3*n,:)), "LineWidth", 1.2); 
title('MC angles for i = 1')
set(gca,'linewidth',1.5, 'fontsize',20); xlabel('t')
xlim([0 tt(end)])


subplot(1,4,4)
plot(tt,squeeze(paths_mc(1,3*n+1:end,:)), "LineWidth", 1.2); 
title('OU noise for i = 1')
set(gca,'linewidth',1.5, 'fontsize',20); xlabel('t')
xlim([0 tt(end)])

% Mean across machines
f2 = figure(2);
f2.Position = [500 500 1500 400];
subplot(1,4,1)
plot(tt,mean(squeeze(paths_mc(1,1:n,:)),1),'linewidth',2,"Color","red"); 
title('Average voltage for i = 1'); xlabel('t')
set(gca,'linewidth',1.5, 'fontsize',20)
xlim([0 tt(end)])

subplot(1,4,2)
plot(tt,mean(squeeze(paths_mc(1,n+1:2*n,:)),1),'linewidth',2,"Color","red"); 
title('Average speeds for i = 1'); xlabel('t')
set(gca,'linewidth',1.5, 'fontsize',20)
xlim([0 tt(end)])

subplot(1,4,3)
plot(tt,mean(squeeze(paths_mc(1,2*n+1:3*n,:)),1),'linewidth',2,"Color","red"); 
title('Average angles for i = 1'); xlabel('t')
set(gca,'linewidth',1.5, 'fontsize',20)
xlim([0 tt(end)])

subplot(1,4,4)
plot(tt,mean(squeeze(paths_mc(1,3*n+1:end,:)),1),'linewidth',2,"Color","red"); 
title('Average OU noise for i = 1'); xlabel('t')
set(gca,'linewidth',1.5, 'fontsize',20)
xlim([0 tt(end)])

%% Compute energy for a specific line
from_line = 8;
to_line = 9;
assert(b(from_line,to_line)~=0.0 | g(from_line,to_line)~=0.0);

% compute energy for each Monte Carlo trial at each time point
mc_energy = zeros(mc,nt);
% compute expectation output for each Monte Carlo trial at each time point
mc_condexp_target = zeros(mc,nt);
for i =1:mc
    i
    for j = 1:nt
        % get solution
        u_i = reshape(paths_mc(i,:,j), [], 1);
        mc_energy(i,j)=line_energy(b,from_line,to_line,u_i);
        mc_condexp_target(i,j)=condexp_target(b,from_line,to_line,u_i,wr);
    end
end

%% Visualize histogram of line energy
for i = 1:nt
    figure(1);
    subplot(1,2,1);
    histogram(mc_energy(:,i),100);
    
    subplot(1,2,2);
    histogram(mc_condexp_target(:,i),100);
end










