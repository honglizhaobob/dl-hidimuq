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

mc = 1000;             % Number of MC paths
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
sd_d = 5.0 * pi/180.0; % sd_d = 10.0 degrees, mean around optimal d0
u0_d = sd_d*randn(mc,n) + reshape(d0,1,[]);

% Random voltages . (folded gaussian, mean at optimal vi)
sd_v = mean(vi)*0.01;
v = abs(sd_v*randn(mc,n) + reshape(vi,1,[]));

% Random initial conditions for OU noise
theta = 0.05;                % drift parameter
alpha = 0.05;               % diffusion parameter

% define covariance matrix
case_number = 9;
mode = "id";
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
assert(b(from_line,to_line)>0.0);

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

%% Kernel density estimation for energy
for i = 1:nt
    [f,xi] = ksdensity(mc_energy(:,i),'Support','positive','BoundaryCorrection','reflection');
    figure(1);
    plot(xi,f,"LineWidth",2.5,"Color","red");
    xlim([0 max(max(mc_energy))]);
    ylim([0 10]);
end

%% Visualize scatter plots of conditional expectation data
n_energy_discretization = 1000;
energy_grid = linspace(0, max(max(mc_energy))-950, n_energy_discretization);
for idx = 1:nt
    figure(1);
    h = scatter(mc_energy(:,idx),mc_condexp_target(:,idx),...
        "MarkerEdgeColor","red");
    h.SizeData = 2;
    % visualize a fitted spline
    fitted_curve = csaps(mc_energy(:,i),mc_condexp_target(:,i),0.1,energy_grid);
    hold on;
    plot(energy_grid, fitted_curve, "Color", "black", "LineWidth", 2.0);
    xlim([0, max(max(mc_energy))]);
    hold off;
    pause(0.1);
end



%% Learn coffcients & solve 1d marginal PDE via lax-wendroff
close all; 
rng('default');
% tic

% Currently solves PDE for marginal PDF of a single fixed oscillator

osc = 5;     % which marginal (oscillator) we want 1<=osc<=N

% Choose angle or velocity marginal pdf for osc
%   = "angle" (delta_i)
%   = "velocity" (w_i)
state = "velocity";

% choose with method for coefficient learning
%   = "llr" for Loc. Lin. (Gaussian) kernel smoothing regress. w/ 10-fold CV
%   = "lin" for standard ordinary least squares linear regression
method = "lin";
              
dx = 0.01;        % cell size for pde
ng = 2;           % number of ghost cells (=2 !! Don't change !!)

% Choose operator splitting (in time) when diffusion is present
split = "godunov"; % ="strang" or "godunov" 

%_____________________________________________________________________

if osc<1 || osc>N || ~isscalar(osc) || rem(osc,1)~=0
    error('Pick integer 1<=osc<=N')
end

if ~(method=="llr" || method=="lin")
    error('Expecting method = llr or lin')
end

hh = H(osc); dd = D(osc); pm = Pm(osc);  % Get associated params ==> see above
si = 3*osc - 2; % index for speed of osc

% get domain for pde
if state == "angle"
    si = si + 1;   % update index
    
    % Padding for PDE domain
    eta = 5*max(std(squeeze(paths_mc(si,:,:)),0,2),[],'all');
    % PDE computational domain:
    a0 = -eta; b0 = eta;
    
elseif state == "velocity"
    
    % Padding for PDE domain
    eta = 4*max(std(squeeze(paths_mc(si,:,:)),0,2),[],'all');
    % PDE computational domain:
    a0 = wr - eta; b0 = wr + eta;
    
    % Diffusion coeff.
    Dco = 0.5*sig(osc)^2;
else
    error('Expecting state = angle or velocity')
end

nx = ceil((b0-a0)/dx);                      % number of nodes
xpts = linspace(a0+0.5*dx, b0-0.5*dx,nx)';  % column vector of cell centers
dx = xpts(2)-xpts(1);                       % updated dx if needed
xpts_e = [xpts-0.5*dx; xpts(end)+0.5*dx];   % cell edges for advection coeff.

% allocate pde solution
f = zeros(nx+2*ng,nt);
f_ind = ng+1:nx+ng;     % indices of non-ghost cells

% IC of PDE via KDE (can generate more samples for IC if needed)
mc2 = 24000;
if state == "angle"
    u0mc = sd_d*randn(mc2,1) + d0(osc);
else
    u0mc = sd_w*randn(mc2,1) + mu_w;
end
    
f0 = [squeeze(paths_mc(si,1,:)); u0mc];
bw = 0.9*min(std(f0), iqr(f0)/1.34)*(mc+mc2)^(-0.2);
f(f_ind,1) = ksdensity(f0,xpts,'bandwidth',bw);
figure(1)
plot(xpts,f(f_ind,1))


% time loop
for nn = 2:nt
    
    % Learn advection coeffcient via data:
   
    % Advection coefficient is time-dependent; currently using it at 
    %   previous time when solving the PDE. 
    
    % The data at previous time
    w = squeeze(paths_mc(3*osc-2,nn-1,:));    % speed for osc
    gov = squeeze(paths_mc(3*osc,nn-1,:));    % governor for osc
    delt = squeeze(paths_mc(2:3:(3*N-1),nn-1,:)); % all angles: N x mc matrix

    % Get adv. coefficient defined on xpts_e (size(coeff) = size(xpts_e))
    coeff = get_coeff(mc,osc,w,gov,delt,R,T1,T2,v,hh,dd,pm,wr,b,g,N,xpts_e,state,method,dx);
    
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
        if state == "angle" % advection only
            f(f_ind,nn) = lax_wen(f(:,nn-1),f_ind,nx,coeff,dx,dt);
        else
            
            % time splitting for advection and diffusion
            if split == "godunov"    % 1st order
                f(f_ind,nn-1) = lax_wen(f(:,nn-1),f_ind,nx,coeff,dx,dt);
                f(f_ind,nn) = diffusion(f(:,nn-1),f_ind,nx,Dco,dx,dt);
                
            elseif split == "strang" % 2nd order
                f(f_ind,nn-1) = lax_wen(f(:,nn-1),f_ind,nx,coeff,dx,dt/2);
                f(f_ind,nn-1) = diffusion(f(:,nn-1),f_ind,nx,Dco,dx,dt);
                f(f_ind,nn) = lax_wen(f(:,nn-1),f_ind,nx,coeff,dx,dt/2);
            else
                error('Expecting split = godunov or strang')
            end
        end
            
        
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
            if state == "angle" % advection only
                f_temp(f_ind) = lax_wen(f_temp,f_ind,nx,coeff,dx,dt2);
            else

                % time splitting for advection and diffusion
                if split == "godunov"    % 1st order
                    f_temp(f_ind) = lax_wen(f_temp,f_ind,nx,coeff,dx,dt2);
                    f_temp(f_ind) = diffusion(f_temp,f_ind,nx,Dco,dx,dt2);

                elseif split == "strang"
                    f_temp(f_ind) = lax_wen(f_temp,f_ind,nx,coeff,dx,dt2/2);
                    f_temp(f_ind) = diffusion(f_temp,f_ind,nx,Dco,dx,dt2);
                    f_temp(f_ind) = lax_wen(f_temp,f_ind,nx,coeff,dx,dt2/2);
                else
                    error('Expecting split = godunov or strang')
                end
            end
        end   
        
        f(f_ind,nn) = f_temp(f_ind);
    end    
    
    if max(abs(f(:,nn)))>1e2
        error('PDE blows up')
    end
    if max(isnan(f(:,nn)))==1
        error('PDE has NaN values')
    end
  
    if mod(nn,50)==0
        disp(nn)
        figure(1)
        set(gca,'linewidth',1.5, 'fontsize',20)
        plot(xpts,f(f_ind,nn),'-b','linewidth',2);
        xlabel('x')
        title('Marginal PDF solutions')
        drawnow
    end
    
end
toc

%_______________________________________________________________________
%_______________________________________________________________________
% Internal functions/subroutines

function coeff = get_coeff(mc,osc,w,gov,delt,R,T1,T2,v,hh,dd,pm,wr,b,g,N,xpts_e,state,method,dx)

    d = delt(osc,:)';
    % Model: yy = r(xx) + e, where E[e] = 0 
    % Under MSE loss, exact solution is r(x0) = E[yy|xx = x0]
    % For us, yy is part of the advection coeff. that can't be pulled out
    %   of the conditional expectation.
    
    if state == "angle"
        xx = d; yy = w;      
    else 
        xx = w;
        
        dmat = repmat(d',N,1) - delt;
        yy = gov - (v(osc,:).*sum(v.*(repmat(g(osc,:)',1,mc).*cos(dmat)...
            + repmat(b(osc,:)',1,mc).*sin(dmat))))';
        % This is only part of the coeff. We don't need to estimate the
        %   other part --> see end of this function.
        
    end
    
    % clean missing or invalid data points, then sort
    if numel(xx) ~= numel(yy)
        error('x and y are in different sizes.');
    end
    xx = xx(:);  yy = yy(:);
    inv = (xx~=xx)|(yy~=yy)|(abs(xx)==Inf)|(abs(yy)==Inf);
    xx(inv)=[];
    yy(inv)=[];
    [xx,xx_ind] = sort(xx); yy = yy(xx_ind);
    
    if  ~isreal(xx) || ~isreal(yy)
        error('Need xx and yy real col. vector')
    end
    
    % number of samples
    ns = length(xx);
   
    nx = length(xpts_e);
    % Allocate coeff
    coeff0 = zeros(nx,1); 
    
    if method == "lin"
        oo = ones(ns,1); X = [oo,xx];
        beta = ((X'*X)\spdiags(oo,0,2,2))*X'*yy;
        coeff0 = beta(2)*xpts_e + beta(1);
    else 
        % method == 'llr'
        % total number of bandwidths choices for CV: logarithmically spaced
        nb = 30; 
        % number of folds for k-fold CV
        kf = 10; 
        % Max Number of padding/extrapolation cells for learning coeff.
        pad = 5; 
        
        % Actual domain for regression (doing extrapolation at +/- pad*dx pts)
        j_ind = find(xpts_e>=(min(xx)-pad*dx) & xpts_e<=max(xx)+pad*dx);
        
        % check inputs for regress_ll function
        if mod(length(xx),kf)~=0 % needs to divide number of samples
            error("Number of CV folds doesn't divide sample size")
        end 
        if nb<2 || rem(nb,1)~=0 || ~isscalar(nb) || ~isreal(nb)
            error('Need (real) integer nb > 1')
        end
        if kf<2 || kf>length(xx) || rem(kf,1)~=0 || ~isscalar(kf) || ~isreal(kf)
            error('Need (real) integer 2 <= kf <= length(xx)')
        end
        coeff0(j_ind) = regress_ll(xx,yy,xpts_e(j_ind),nb,kf,dx,state);
        
        % linear extrapolation
        mx1 = (coeff0(j_ind(2))-coeff0(j_ind(1)))...
                /(xpts_e(j_ind(2))-xpts_e(j_ind(1)));
        b1 = coeff0(j_ind(1)) - mx1*xpts_e(j_ind(1));
        coeff0(1:j_ind(1)-1) = mx1*xpts_e(1:j_ind(1)-1) + b1;

        mx2 = (coeff0(j_ind(end))-coeff0(j_ind(end-1)))...
               /(xpts_e(j_ind(end))-xpts_e(j_ind(end-1)));
        b2 = coeff0(j_ind(end)) - mx2*xpts_e(j_ind(end));
        coeff0(j_ind(end)+1:end) = mx2*xpts_e(j_ind(end)+1:end) + b2;
        
    end
    
    % check is nan
    if max(isnan(coeff0))==1
        error('coeff0 is NaN in get_coeff()')
    end
                        
    % Uncomment to plot coeff0
 
%     figure(2)
%     plot(xx,yy,'ko','markersize',2); hold on;
%     plot(xpts_e,coeff0,'-b','linewidth',2); 
%     set(gca,'linewidth',1.5, 'fontsize',20)
%     title('Coeff0 Estimate'); xlabel('X_5'); ylabel('f(X_5,t)'); 
% %     xlim([min(xx,[],'all') max(xx,[],'all')]);
%     hold off; drawnow
    
    if state == "angle"
        coeff = coeff0 - wr;
    else
        coeff = 0.5*wr*(pm - (dd + T1/(R*T2))*(xpts_e-wr) + coeff0)/hh;
        
%         figure(3)
%         plot(xpts_e,coeff,'-b','linewidth',2); 
%         set(gca,'linewidth',1.5, 'fontsize',20)
%         title('Vel. Advection Coeff.'); xlabel('X_5'); ylabel('f(X_5,t)'); 
%         drawnow
    end
end
%_______________________________________________________________________

function coeff0 = regress_ll(xx,yy,xpts,nb,kf,dx,state)
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
      if state == "angle"
          bw_min = 3*dx;   
          %%% WARNING! MANUALLY TUNED 
          %%% may need to be chnaged for different case
      else
          bw_min = 2*dx;
      end
      
      % nb possible bandwidths
      % log spaced 
      bw = exp(linspace(log(bw_min),log(max(xx)-min(xx)),nb));
%       % linearly spaced + plug-in:
%       bw = sort([r0.h, linspace(dx,max(xx)-min(xx),nb-1)]); 

    %  k-fold CV for bandwidth selection
    [kscv_err, kscv_se] = ksrlin_cv(xx,yy,bw,nb,kf);

    % optimal bandwidth
    [~, ksmin_ind] = min(kscv_err); 
%     if ksmin_ind == nb
%         warning('CV minimized at maximal bw --> increase bandwidth range')
%     elseif ksmin_ind == 1
%         warning('CV minimized at minmal bw --> increase bandwidth range')
%     end
    
%     errors0 = [sqrt(ksmin),kscv_se(ksmin_ind)];
    
    % 1 St. Err. rule of thumb (increase regularity)
%     bw_1se = bw(find(bw(ksmin_ind:end) <= bw(ksmin_ind)+kscv_se(ksmin_ind),...
%                  1, 'last' ) - 1 + ksmin_ind);

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

%_______________________________________________________________________

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