%     Hongli Zhao, Givens Associate, Argonne Nat. Lab.,
%     CAM PhD Student, UChicago
%     Last updated: Aug 10, 2023
%% Define simulation parameters
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

%% Manually remove a line
% need to figure out how to remove a line ... proceeding without manual
% removal

%% Sanity check: simulate deterministic equations with optimal initial conditions
% initial condition
x0 = zeros(3*n,1);
% initial velocities
x0(1:n) = vi;
% initial speed
x0(n+1:2*n) = ones(n,1);
% initial angles
x0(2*n+1:end) = zeros(n,1);
[~, x_sol] = ode23(@(t,x) rhs2(x,H,D,Pm,wr,g,b), time, x0);
x_sol = x_sol';

% visualize deterministic solutions
f = figure(1);
f.Position = [500 500 1240 400];
subplot(1,3,1)
plot(tt,x_sol(1:n,:), "LineWidth", 1.2); 
title('Voltages')
set(gca,'linewidth',1.5, 'fontsize',20); xlabel('t')
xlim([0 tt(end)])

subplot(1,3,2)
plot(tt,x_sol(n+1:2*n,:), "LineWidth", 1.2); 
title('Speeds')
set(gca,'linewidth',1.5, 'fontsize',20); xlabel('t')
xlim([0 tt(end)])

subplot(1,3,3)
plot(tt,x_sol(2*n+1:3*n,:), "LineWidth", 1.2); 
title('Angles')
set(gca,'linewidth',1.5, 'fontsize',20); xlabel('t')
xlim([0 tt(end)])


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
%% Save data
save("./data/case9_mc.mat");
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

%% Learn coffcients & solve 1d marginal PDE via lax-wendroff
close all; 
rng('default');

% line is fixed and data simulated above
from_line; to_line; mc_energy; mc_condexp_target;
% set up pde domain
dx = 0.1;          % spatial step size
ng = 2;             % number of ghost cells
a0 = -1.0;            % energy cannot be negative
b0 = max(mc_energy,[],"all")+1.0*max(std(mc_energy)); % padded right boundary
nx = ceil((b0-a0)/dx);                      % number of nodes
xpts = linspace(a0+0.5*dx, b0-0.5*dx,nx)';  % column vector of cell centers
dx = xpts(2)-xpts(1);                       % updated dx if needed
xpts_e = [xpts-0.5*dx; xpts(end)+0.5*dx];   % cell edges for advection coeff

% allocate pde solution (IC from kernel density estimate)
f = zeros(nx+2*ng,nt);
f_ind = ng+1:nx+ng;     % indices of non-ghost cells

    
f0 = [squeeze(mc_energy(:,1))];
bw = 0.9*min(std(f0), iqr(f0)/1.34)*(mc)^(-0.2);
f(f_ind,1) = ksdensity(f0,xpts,'bandwidth',bw,'Support','positive', ...
    'BoundaryCorrection','reflection');
figure(1);
plot(xpts, f(f_ind,1),"LineWidth",1.5,"Color","black");

%%

all_first_moments_ropdf = [];
all_first_moments_ground_truth = [];
all_second_moments_ropdf = [];
all_second_moments_ground_truth = [];
% time loop
for nn = 2:nt
    if nn>=200
        break;
    end
    % learn advection coeffcient via data and propagate PDE dynamics
    % Exact solution is of form: E[Y | X]

    % get X data (previous time)
    energy_data = squeeze(mc_energy(:,nn-1));
    
    % get Y data (previous time)
    response_data = squeeze(mc_condexp_target(:,nn-1));

    % compute advection coefficient (need to be defined on cell centers)

    % Get adv. coefficient defined on xpts_e (size(coeff) = size(xpts_e))
    coeff = get_coeff(energy_data,response_data,xpts_e,"lin",alpha,theta,C);

    % for all coeffs outside of the main support, set to 0, technically
    % undefined (any locations outside of max of energy
    % observed)
    tmp = find((xpts_e>max(energy_data)-3*std(energy_data)));
    coeff(tmp) = 0.0;
    % CFL condition for Lax-Wendroff --> variable time stepping
    u_mag = max(abs(coeff));
    if u_mag==0
        dt2 = dt;
    else
        % CFL
        dt2 = dx/u_mag; 
    end
    
    if dt2 >= dt  % use the dt we already had
        % Homogeneous Dirchlet BC's already set from allocation
        f(f_ind,nn) = lax_wen(f(:,nn-1),f_ind,nx,coeff,dx,dt);
%         else
%             
%             % time splitting for advection and diffusion
%             if split == "godunov"    % 1st order
%                 f(f_ind,nn-1) = lax_wen(f(:,nn-1),f_ind,nx,coeff,dx,dt);
%                 f(f_ind,nn) = diffusion(f(:,nn-1),f_ind,nx,Dco,dx,dt);
%                 
%             elseif split == "strang" % 2nd order
%                 f(f_ind,nn-1) = lax_wen(f(:,nn-1),f_ind,nx,coeff,dx,dt/2);
%                 f(f_ind,nn-1) = diffusion(f(:,nn-1),f_ind,nx,Dco,dx,dt);
%                 f(f_ind,nn) = lax_wen(f(:,nn-1),f_ind,nx,coeff,dx,dt/2);
%             else
%                 error('Expecting split = godunov or strang')
%             end
%         end
            
        
    else
        % CFL time step is smaller than time step for the samples.
        % Solve pde at intermediate times with smaller time step,
        %   using same coeff., and then output solution on the coarser time
        %   grid.
        
        nt_temp = ceil(dt/dt2)+1; 
        dt2 = dt/(nt_temp - 1);
        f_temp = f(:,nn-1);
        
        if dt2==0 || isnan(dt2)
            error('dt0 = 0 or NaN')
        end
        
        for ll = 2:nt_temp
            f_temp(f_ind) = lax_wen(f_temp,f_ind,nx,coeff,dx,dt2);
%             else
% 
%                 % time splitting for advection and diffusion
%                 if split == "godunov"    % 1st order
%                     f_temp(f_ind) = lax_wen(f_temp,f_ind,nx,coeff,dx,dt2);
%                     f_temp(f_ind) = diffusion(f_temp,f_ind,nx,Dco,dx,dt2);
% 
%                 elseif split == "strang"
%                     f_temp(f_ind) = lax_wen(f_temp,f_ind,nx,coeff,dx,dt2/2);
%                     f_temp(f_ind) = diffusion(f_temp,f_ind,nx,Dco,dx,dt2);
%                     f_temp(f_ind) = lax_wen(f_temp,f_ind,nx,coeff,dx,dt2/2);
%                 else
%                     error('Expecting split = godunov or strang')
%                 end
%             end
        end   
        
        f(f_ind,nn) = f_temp(f_ind);
    end    
    
    if max(abs(f(:,nn)))>1e2
        error('PDE blows up')
    end
    if max(isnan(f(:,nn)))==1
        error('PDE has NaN values')
    end
    disp(nn)
    % visualize learned coeffs, RO-PDF solution and exact solution from KDE
    if mod(nn,1)==0
        fig=figure(1);
        fig.Position = [100 500 1600 400];

        % Learned coefficients and data
        subplot(1,4,1);
        scatter(mc_energy(:,nn),mc_condexp_target(:,nn),...
            "MarkerEdgeColor","red","SizeData",2.0);
        hold on;
        plot(xpts_e,coeff,"--","LineWidth",2.5,"Color","black");
        hold off;
        title('Learned Coefficients');
        xlabel('Line Energy');
        

        % RO-PDF predictions
        subplot(1,4,2);
        set(gca,'linewidth',1.5, 'fontsize',20)
        f_pred = f(f_ind,nn);
        mass = trapz(dx,f_pred);
        f_pred=f_pred/mass;
        disp(mass);
        
        plot(xpts,f_pred,'-b','linewidth',2);
        xlabel('x')
        title('Marginal PDF solutions');

        % KDE ground truth
        subplot(1,4,3);
        set(gca,'linewidth',1.5, 'fontsize',20)
        f0 = [squeeze(mc_energy(:,nn))];
        %bw = 0.9*min(std(f0), iqr(f0)/1.34)*(mc)^(-0.2);
        f_kde = ksdensity(f0,xpts,'Support','positive', ...
            'BoundaryCorrection','reflection');
        mass=trapz(dx,f_kde);
        disp(mass)
        
        all_first_moments_ropdf = [all_first_moments_ropdf trapz(dx,xpts.*f_pred)];
        all_first_moments_ground_truth = [all_first_moments_ground_truth trapz(dx,xpts.*f_kde)];
        all_second_moments_ropdf = [all_second_moments_ropdf trapz(dx,(xpts.^2).*f_pred)];
        all_second_moments_ground_truth = [all_second_moments_ground_truth trapz(dx,(xpts.^2).*f_kde)];

        f_kde=f_kde/mass;
        plot(xpts,f_kde,"LineWidth",1.5,"Color","black");

        % compute pointwise error
        subplot(1,4,4);
        plot(xpts,abs(f_kde-f_pred),"LineWidth",1.5,"Color","black");
    end
end
toc

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function coeff0 = get_coeff(xx, yy, xpts_e, mode, alpha, theta, C)
    % Computes advection coefficient for line energy marginal (on cell 
    % centers) based on regression from scatter data. 
    %
    % Model: yy = r(xx) + error, where E[error] = 0
    %
    % Exact solution is r(x0) = E[yy|xx = x0]
    if numel(xx)~=numel(yy)
        error("x and y are of different sizes. ");
    end
    xx = xx(:); yy = yy(:);
    % sort the data
    [xx,xx_ind] = sort(xx); yy = yy(xx_ind);
    % number of samples
    ns = length(xx);
    nx = length(xpts_e);
    dx = xpts_e(2)-xpts_e(1);
    if mode == "const"
        % constant regression (simply the average)
        coeff0 = ones(nx,1).*mean(yy);
    elseif mode=="lin"
        % linear regression (parameteric)
        oo = ones(ns,1); X = [oo,xx]; % include bias term
        beta = ((X'*X)\spdiags(oo,0,2,2))*X'*yy;
        coeff0 = beta(2)*xpts_e + beta(1);
    elseif mode == "llr"
        % local linear regression (nonparameteric)
        coeff0 = zeros(nx,1);
        % total number of bandwidths choices for CV: logarithmically spaced
        nb = 30; 
        % number of folds for k-fold CV
        kf = 10; 
        % Max Number of padding/extrapolation cells for learning coeff.
        pad = 2; 
        % Actual domain for regression (doing extrapolation at +/- pad*dx pts)
        j_ind = find(xpts_e>=(min(xx)-pad*dx) & xpts_e<=max(xx)+pad*dx);
        coeff0(j_ind) = regress_ll(xx,yy,xpts_e(j_ind),nb,kf,dx);

        % linearly extrapolate
        mx1 = (coeff0(j_ind(2))-coeff0(j_ind(1)))...
                 /(xpts_e(j_ind(2))-xpts_e(j_ind(1)));
         b1 = coeff0(j_ind(1)) - mx1*xpts_e(j_ind(1));
         coeff0(1:j_ind(1)-1) = mx1*xpts_e(1:j_ind(1)-1) + b1;
    
         mx2 = (coeff0(j_ind(end))-coeff0(j_ind(end-1)))...
                /(xpts_e(j_ind(end))-xpts_e(j_ind(end-1)));
         b2 = coeff0(j_ind(end)) - mx2*xpts_e(j_ind(end));
         coeff0(j_ind(end)+1:end) = mx2*xpts_e(j_ind(end)+1:end) + b2;

    elseif mode == "nn"
        % neural network prediction (considered nonparameteric)
        error("not implemented. ");
    else
        error("mode undefined.");
    end

    % check if any is nan and throw error
    if any(isnan(coeff0))
        error('coeff0 is NaN in get_coeff()')
    end

    % modify learned coefficient to include analytic terms, which is
    % a constant for line energy
    coeff0 = coeff0 - 2*(alpha^2)*(theta^2)*trace(C'*C);
end

function coeff0 = regress_ll(xx,yy,xpts,nb,kf,dx)
    %  Local Linear (Gaussian) Kernel Regression with kf-fold bandwidth selection
    %  approximates f(x) in model Y = f(X) + e, where E[e] = 0
    
    % Input:
    %   xx := column vector of x data 
    %   yy := f(xx) + e
    %   xpts := query points to fit regression (column vector)
    %   nb>1 := total number of bandwidths for k-fold CV
    %   1<kf<=length(xx) := number of folds for CV 
    
    % Output:
    %   coeff0 := estimates regression function f(xpts) 

    % Optimal asymptotic bandwidth ==> assumes e ~ normal
%     r0 = ksrlin(xx,yy); 
    
    % nb possible bandwidths
      % log spaced 
      bw_min = 5*dx; %%% WARNING! MANUALLY TUNED
      
      % nb possible bandwidths
      % log spaced 
      bw = exp(linspace(log(bw_min),log(max(xx)-min(xx)),nb));
%       % linearly spaced + plug-in:
%       bw = sort([r0.h, linspace(dx,max(xx)-min(xx),nb-1)]); 

    %  k-fold CV for bandwidth selection
    [kscv_err, kscv_se] = ksrlin_cv(xx,yy,bw,nb,kf);

    % optimal bandwidth
    [~, ksmin_ind] = min(kscv_err); 
    bw_1se = bw(ksmin_ind) + kscv_se(ksmin_ind);
    r_opt = ksrlin(xx,yy,bw_1se,xpts);   
    coeff0 = r_opt.f; 
end
%_______________________________________________________________________
function [cv_err, cv_se] = ksrlin_cv(xi,yj,bw,nb,kk)

    % kk-fold CV for local linear kernel smoothing regression
    nsize = numel(xi);
    cv_part = cvpartition(nsize, 'kfold',kk);
    ntest = cv_part.TestSize(1); 
    cv_err = zeros(1,nb); cv_se = cv_err;

    parfor ii = 1:nb

        % response data for current bw
%         r = ksrlin(xi,yj,bw(ii),xpts);

        % loop over folds
        cv_k = zeros(ntest,kk); 
        for jj = 1:kk
            
            test_ind = cv_part.test(jj);
            train_ind = cv_part.training(jj);

            % train/test 
            Y_te = ksrlin(xi(train_ind),yj(train_ind)...
                           ,bw(ii),xi(test_ind));

            cv_k(:,jj) = (yj(test_ind)-Y_te.f).^2;   
        end
        cv_err(ii) = mean(cv_k,'all');
        cv_se(ii) = sqrt(sum((cv_k - cv_err(ii)).^2,'all')/(ntest*(ntest-1)));
    end
    
end

function ff = lax_wen(f,f_ind,nx,u,dx,dt)

    % Takes one time step of 1d conservative advection equation
    %   via lax-wendroff with MC limiter
    
    % Reference: LeVeque, Randall J. Finite volume methods for hyperbolic problems.
    %   Vol. 31. Cambridge university press, 2002.
    
    % Input:
    %   f := solution at current times step, (nx+2*ng,1) vector
    %   f_ind := indices of f of non-ghost cells
    %   nx := number of non-ghost cells
    %   u := variable advec. coeff., (nx+1,1) vector 
    %        defined on left cell edges (1-1/2):(nx+1/2)
    %   dt and dx are time and spatial step
    
    % Output: 
    %   f0 := (nx,1) solution at next time step on non-ghost cells
    
    % Positive and negative speeds
    indp = find(u>0); indm = find(u<0);
    up = zeros(nx+1,1); um = up;
    up(indp) = u(indp); um(indm) = u(indm);
    
    % 1st-order right and left going flux differences
    % LeVeque sect. 9.5.2 The conservative equation

    % At cell i: Apdq(i-1/2) = right going  flux = F(i) - F(i-1/2),
    %            Amdq(i+1/2) = left going  flux  = F(i+1/2) - F(i),
    %            where F is numerical flux.
    % Upwind edge flux: F(i-1/2) = up(i-1/2)f(i-1) + um(i-1/2)f(i),
    %                   F(i-1/2) = up(i-1/2)f(i-1) + um(i-1/2)f(i).
    % Cell flux can be taken arbitrarily, i.e. F(i) = 0,
    %   but more asthetic to approximte it:
    %   F(i) = (up(i-1/2) + um(i+1/2)*f(i).
    
    % Apdq(i-1/2)= F(i) - F(i-1/2),  Amdq(i+1/2) = F(i+1/2) - F(i);
    % F(i-1/2) = up(i-1/2)f(i-1) + um(i-1/2)f(i)
    % F(i+1/2) = up(i+1/2)f(i) + um(i+1/2)f(i+1)
    
    % F(i)
%     Fi = (up(f_ind) + um(f_ind+1)).*f(f_ind);
    Flux_i = 0;
    % F(i-1/2)
    Flux_m = up(1:nx).*f(f_ind-1) + um(1:nx).*f(f_ind);
    % F(i+1/2)
    Flux_p = up(2:nx+1).*f(f_ind) + um(2:nx+1).*f(f_ind+1);
    % Apdq(i-1/2) and Amdq(i+1/2)
    Apdq = Flux_i - Flux_m;  Amdq = Flux_p - Flux_i;

    % W = wave with speed u; p = i+1/2, m = i-1/2
    Wp = f(f_ind+1) - f(f_ind); Wm = f(f_ind) - f(f_ind-1);

    % theta's for limiter: LeVeque book sect. 9.13
    % theta_i-1/2 = q(I) - q(I-1) / Wm , I = i-1 u_i-1/2>=0, =i+1 u_i-1/2<0
    % theta_i+1/2 = q(I+1) - q(I) / Wp , I = i-1 u_i+1/2>=0, =i+1 u_i+1/2<0
    
    % Allocate for limiters
    Thm =  zeros(nx,1); Thp = Thm;
    
    % At i-1/2
    xsm = indm(indm<nx+1); xsp = indp(indp<nx+1);
    Thm(xsm) = (f(f_ind(xsm)+1) - f(f_ind(xsm)))./Wm(xsm);     % negative speed
    Thm(xsp) = (f(f_ind(xsp)-1) - f(f_ind(xsp)-2))./Wm(xsp);   % positive speed
    
    % At i+1/2
    xsm = indm(indm>1)-1; xsp = indp(indp>1)-1;
    Thp(xsm) = (f(f_ind(xsm)+2) - f(f_ind(xsm)+1))./Wp(xsm);     % negative speed
    Thp(xsp) = (f(f_ind(xsp)) - f(f_ind(xsp)-1))./Wp(xsp);   % positive speed
    
    % MC limiter: LeVeque sect. 6.12 TVD Limiters eqn (6.39a)
    phip = max(0,min(min((1+Thp)/2,2),2*Thp));
    phim = max(0,min(min((1+Thm)/2,2),2*Thm));
    
    % mW = modified wave, LeVeque sect. 9.13 eqn (9.69)
    mWp = phip.*Wp; mWm = phim.*Wm;
      
    % 2nd-order limited corrections: LeVeque sect. 6.15 eqn (6.60)
    Fp = 0.5*abs(u(2:nx+1)).*(1 - (dt/dx)*abs(u(2:nx+1))).*mWp;
    Fm = 0.5*abs(u(1:nx)).*(1 - (dt/dx)*abs(u(1:nx))).*mWm;
    
    ff = f(f_ind) - (dt/dx)*(Apdq + Amdq + Fp - Fm);
    
    if ~isequal(size(ff),[nx,1])
        error('lax_wen output has incorrect size')
    end

end

%_______________________________________________________________________

function ff = diffusion(f,f_ind,nx,D,dx,dt)

    % Takes one time step of 1d heat equation f_t = D*f_xx
    %   via Crank Nicolson and central differencing
    
    % Input:
    %   f := solution at current times step, (nx+2*ng,1) vector
    %   f_ind := indices of f of non-ghost cells
    %   nx := number of non-ghost cells
    %   D := scalar diffusion coefficient
    %   dt and dx are time and spatial step
    
    % Output: 
    %   f0 := (nx,1) solution at next time step on non-ghost cells
    
    dtdx2 = dt/(dx*dx);
    
    % Crank-Nicolson Matrix --> homogenous Dirichelt BCs
    oo = ones(nx,1);
    M = spdiags([-0.5*D*dtdx2*oo, (1+D*dtdx2)*oo, -0.5*D*dtdx2*oo],...
        -1:1,nx,nx);
    
    ff = M\(f(f_ind) + 0.5*D*dtdx2*(f(f_ind-1) - 2*f(f_ind) + f(f_ind+1)));
    
    if ~isequal(size(ff),[nx,1])
        error('ldiffusion output has incorrect size')
    end

end