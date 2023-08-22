%     PROJECT: Data-driven marginal PDF equations for Langevin systems
%    
%     AUTHOR: Tyler E Maltba

%     AFFILIATIONS: 
%        Argonne National Lab: MCS Division
%                   Positions: Givens Associate, Research Aide
%                 Supervisors: Daniel Adrian Maldonado, Vishwas Rao
%
%        UC Berkeley: Dept. of Statistics
%          Positions: PhD Candidate, NSF Graduate Research Fellow

%     LAST UPDATE: OCT. 29, 2021
              
%     DESCRIPTION:
%        We consider the reduced-order PDF method approach applied to a
%        multimachine classical power model with N machines/oscillators. 
%        The dynamics of each oscillator is governed by two equations: one
%        for speed/velocity, and one for angle/phase.
%
%        Some code from MatPower 7.1 as been extracted and used to export 
%        admittance matrices and paramaters (e.g., equilibrium voltages, 
%        power bus injections, etc.) from IEEE test cases.
%
%        We have incorporated correlated OU processes for the
%        speeds and angles.

%     DIRECTORY: ~/MATLAB/MATLAB_code\ANL-Givens 2021/Simple Power/
%                 Parallel RO
%                 
%     NEEDED FILES:
%        matpower_addpath_commands.m (WARNING: depends on local path)
%        runpf_edited.m
%        classical_OU.m
%        get_coeff.m
%        regress_ll.m
%        ksrlin.m
%        lax_wen.m


%     Simpliefied NON-ADAPTIVE VERSION:
%        Generates SDE samples ahead of time outside time loop for PDE.
%        ICs are also found via MC
%--------------------------------------------------------------------------

clear; clear; close all
rng('default')       % For reproducibility
tic                  % Start first wall clock

%------------------------------------------------------------------------
mc = 1000;           % Number of MC samples of SDE at each time step 
                     %   for coeffcient learning.

tf = 10;              % final time for SDE simulation and PDE
dt = 0.005;           % SDE & PDE time step 
t = (0:dt:tf)';     % PDE time grid 
nt = length(t); 

%--------------------------------------------------------------------------
% Make sure Matpower files added to path 
matpower_addpath_commands

% choose case from matpower7.1/data
[branch,Pm,Ybus,vi,d0,success] = runpf_edited('case9.m');

if success ~= 1
    error('runpf_edited did not run successfully')
end

% MatPower outputs values for the system in equilibrium:
% dist_mat = (_x4) matrix of branch data
% Pm = vector of total bus power injections
% Ybus = complex admittance matrix
% vi = voltage magnitudes
% d0 = initial angles delta(0) (radians)

% line fail
Ybus(8,6) = 0;  Ybus(6,8) = 0;

% admittance matrices
g = real(Ybus); % for cosines (generators)
b = imag(Ybus); % for sines (buses)

% ODE model parameters
N = length(d0);       % Number of machines/buses
N_ones = ones(N,1);   % Used throughout
H = N_ones;           % Inertia coefficients
D = N_ones;           % Damping coefficients
wr = N_ones;          % Equilibrium speed

% Number of model and noise equations per oscillator/machine
neq = 2;             % number of state eqns (e.g., speed, angle)
nxi  = 1;            %    nxi = 1 --> Only speeds have noise
                     %    nxi = 2 --> speeds and angles have noise
                     
% States and noise will be store in different arrays
% Number of states  = neq*N,  Number of noise = nxi*N
% States are ordered 1:N = speeds, N+1:2*N = angles
% Noises are ordered same as states if nxi>1

%%%%%%%%%%%%%%
% Choose whether the noise processes are uncorrelated
correlation = "none";
% = "none", "exponential", "inverse", "constant"

% Percentage of equilibriums for st. devs. 
sd_ratio = 0.05*ones(neq,1); % (5%)
sd_ratio = [sd_ratio; 0.05];   % add voltage component on the end (5%)
   % governors and noise not included
   % sd_ratio(1) --> for speeds
   % sd_ratio(2) --> for angles if applicable
   % sd_ratio(end) --> for voltages

% Choose with method for coefficient learning
%   = "llr" for Loc. Lin. (Gaussian) kernel smoothing regress. w/ 10-fold CV
%   = "lin" for standard ordinary least squares linear regression
method = "llr";

% Choose marginal (state) we want 1<=state_vec<=neq*N, vector
state = 1; 

%_____________________________________________________________________
% Check inputs
if neq<=1 || neq>2 || nxi<0 || nxi>neq
    error('Check model constants')
end
if ~(isscalar(state) && isreal(state) && state>=1 && state<=neq*N)
    error('Check states')
end
if ~(method=="llr" || method=="lin")
    error('Expecting method = llr or lin')
end

%--------------------------------------------------------------------------
% Correlated OU Noise Parameters and ICs: 

% Correlation Matricies:  Manually set for now only for speeds
if nxi > 1
    error('Rho only set for nxi = 1 case')
end

if correlation == "none"
    % only speeds of same machine are correlated
    Rww = eye(N);
else
    
    % compute branch distances
    br_x = branch(:,3);  br_b = branch(:,4);

    % br_b can't = 0 --> using linear extrap --> case dependent
%     indbr = find(br_b ~= 0);
%     br_b(br_b==0) = interp1(br_x(indbr),br_b(indbr),br_x(br_b==0),'linear','extrap');
    br_b(br_b==0) = min(br_b(br_b>0));

     % physical branch distance matrix (miles)
    dist = 494*sqrt(br_x.*br_b);
    dist_ind1 = [branch(:,1);branch(:,2)];     % row indices
    dist_ind2 = [branch(:,2);branch(:,1)];     % column indicies
    dist_global = N*(dist_ind2-1) + dist_ind1; % global indices
    dist_mat = sparse(dist_ind1, dist_ind2, repmat(dist,2,1), N, N);
    dist_mat = full(dist_mat);
    
    if correlation == "exponential"
        % exponentially correlated based on distance
        if N==9
            length = 82;
        elseif N==30
            length = 14.5;
        end
        Rww = exp(-dist_mat/length);
    elseif correlation == "inverse"
        % inversely proportionally correlated
        if N==9
            length = 53;
        elseif N==30
            length = 2.5;
        end
        Rww = 1./(dist_mat/length + 1);
    elseif correlation == "constant"
        if N==9
            length = 0.44;
        elseif N==30
            length = 0.36;
        end
        Rww = zeros(N,N);
        Rww(dist_mat>0) = length;
        Rww(1:N+1:N^2) = 1;
    else
        error('Check correlation')
    end
    
    % Updated gloabal indices to include diagonal
    dist_global = [dist_global;(1:N+1:N^2)'];
    % correlation matrix for speeds
    Rww(~ismember(1:N^2,dist_global)) = 0;
end

Rwd = [];       % Cross correlation matrix of speeds and angles
Rdd = [];       % Correlation matrix of angles

Rho = [Rww, Rwd; Rwd', Rdd];
% min(eig(Rho))
C = chol(Rho,'lower');

% Noise amplitudes --> % st. dev. for OU noise distribution

if nxi == 1
    sig = sd_ratio(1)*wr;  % percent of equilibrium value
elseif nxi == 2
    % for angles
    sig_d = abs(d0)*sd_ratio(2);   % percent of equilibrium value
    sig_d(d0==0) = mean(d0)*sd_ratio(2); 
    % [speeds, angles]
    sig = [sd_ratio(1)*wr; sig_d];
else
    error('sig only set for nxi = 1 or 2 cases')
end

% th = 1/correlation times
th =  1*repmat(N_ones,nxi,1);

% Volatility Matrix
vol = repmat(sqrt(2*th).*sig,1,nxi*N).*C;
% Covariance of stationary OU  = cov*cov'
cov = chol(sylvester(diag(th),diag(th),vol*vol'),'lower');
% Stationary ICs for noise
xi00 = cov*randn(nxi*N,mc);
xi0 = cov*randn(nxi*N,mc);

vol = sparse(vol);

%--------------------------------------------------------------------------
% Get data to compute random ICs via MC and KDE:

% MC realizations of vi (folded gaussian) --> (N x mc matrix)
sd_v = abs(vi)*sd_ratio(end);    % st. dev--> percent of equilibrium
sd_v(vi==0) = abs(mean(vi))*sd_ratio(end);
v = get_v(N,mc,vi,sd_v);         % size = n x mc0

% deterministic ICs for states
u00 = [wr; d0];  u00 = repmat(u00,1,mc);

[u_eq, xi_eq] = classical_OU(mc,nt,dt,u00,N,H,D,wr,Pm,g,b,v,nxi,xi00,th,vol);

% plot paths for single machine --> first argument indicates the machine
% path_plot(1,N,t(1:end-1)+dt/2,u_eq,xi_eq)

clockIC1 = toc; disp('IC Clock 1: '); disp(clockIC1)

%_____________________________________________________________________
% Compute MC with random ICs
tic

% We want ICs to be previous data at last time (i.e., equilibrium) 
u0 = squeeze(u_eq(:,end,:));

% MC paths of states (u) and noise (xi)
% Outputs at times (dt/2):dt:(nt*dt - (dt/2))
[u, xi] = classical_OU(mc,nt-1,dt,u0,N,H,D,wr,Pm,g,b,v,nxi,xi0,th,vol);

% plot paths for single machine
path_plot(1,N,t(1:end-1)+dt/2,u,xi)

clockIC2 = toc; disp('IC Clock 2: '); disp(clockIC2)

%% Compute domain for PDE based on MC paths
tic

u_st = [u0(state,:); squeeze(u(state,:,:))];  % paths for state of interest

% PDE grid size via Silverman's rule
sdev = std(u_st,0,2);   % sample st. devs. of paths
bw_temp = 0.9*min(sdev,iqr(u_st,2)/1.34)*mc^(-0.2);
dx = min(bw_temp);  % choose smallest over all times

% Padding PDE domain
eta = max(sdev);
% PDE Compuatational boundaries
a0 = min(u_st,[],'all') - eta; 
b0 = max(u_st,[],'all') + eta;

b_length = b0 - a0; 
if b_length <= 0 || isnan(b_length) || abs(b_length) == Inf
   error('b_length <= 0, or = NaN, Inf')
end
if dx >= b_length || dx == 0 || abs(dx) == Inf || isnan(dx)
    error('dx >= b_length, or = 0, NaN, Inf')
end 

% number of cells
nx = ceil(b_length/dx); 
if nx <= 0 || isnan(nx) || abs(nx) == Inf
    error('nx <= 0, or = NaN, Inf')
end
dx = b_length/nx;

% For computing IC via KDE
bw = bw_temp(1);  
if isnan(bw) || bw == Inf
    error('bw = NaN, Inf')
end

% Spatial grid
xpts_e = linspace(a0,b0,nx+1)';  % cell edges 
xpts = xpts_e(1:nx) + dx/2;        % cell centers

clock_dom = toc; disp('Domain Clock: '); disp(clock_dom)

%% Advection coeffs
tic

% which oscillator/machine
osc = mod(state,N);  
if osc == 0 % correct when divisble by N
    osc = N;   
end

% Get associated params H, D, Pm, and wr for current machine
hh = H(osc); dd = D(osc); pm = Pm(osc); wwr = wr(osc);

% Allocate
advect = zeros(nx+1,nt-1);
CFLs = zeros(nt-1,1);
 
% Initialize
t0 = -dt/2;
    
for nk = 1:nt-1
    t0 = t0 + dt;
    
  % data at midpoint in time 
   u_n = squeeze(u(:,nk,:)); xi_n = squeeze(xi(:,nk,:)); 

  % Calculate advection coefficient at for next time
  %    defined on xpts_e (size(coeff) = size(xpts_e)) 
  [coeff, coemax] = get_coeff(nk,method,mc,N,osc,state,u_n,xi_n,nxi,v,hh,dd,pm,wwr,b,g,xpts_e,dx);
  CFLs(nk,1)  = dt*coemax/dx;
  advect(:,nk) = coeff;
          
end

clock_coeff = toc; disp('Coefficient Clock: '); disp(clock_coeff)

[CFLmax, CFLind] = max(CFLs);
if CFLmax > 1
    dt_new = 0.9*dt/CFLmax; %#ok<NASGU>
    error('CFL violated! Rerun SDE simulations with time step = dt_new')
end

%% Solve 1d PDEs in parallel via lax-wendroff with MC limiter
tic 

% Allocate Solution Array
ng = 2; % Number of ghost cells per boundary !!!!! DO NOT CHANGE !!!!!!
f = zeros(nx+2*ng,nt);
f_ind = ng+1:nx+ng;     % indices of non-ghost cells

% IC of PDE via KDE 
f(f_ind,1) = ksdensity(u_st(1,:)',xpts,'bandwidth',bw);

figure(2)
plot(xpts,f(f_ind,1))
      
% time loop of pde
for nn = 2:nt 

   f(f_ind,nn) = lax_wen(f(:,nn-1),f_ind,nx,advect(:,nn-1),dx,dt);

%    if max(f(f_ind,nn)) > 10*max(f(f_ind,1))
%        error('PDE blows up')
%    end
%    if max(isnan(f(f_ind,nn)))==1
%        error('PDE has NaN values')
%    end
   if mod(nn,20)==0
       figure(3)
       plot(xpts,f(f_ind,nn))
       drawnow
   end
end         
    
clock_pde = toc; disp('PDE Clock: '); disp(clock_pde)

bweq = 0.9*min(std(u_st(end,:)),iqr(u_st(end,:))/1.34)*(mc^(-0.2));
feq = ksdensity(u_st(end,:)',xpts,'bandwidth',bweq);
L2 = sqrt(trapz(xpts,(feq-f(f_ind,end)).^2)/trapz(xpts,feq.^2))

%_______________________________________________________________________
%_______________________________________________________________________
%% Internal functions/subroutines

function v = get_v(N,mc,vi,sd_v)

    % INPUT:
    % N     (scalar): number of oscillators
    % mc    (scalar): number of MC trials
    % vi    (N x 1 vec): equilibrium voltage for each machine
    % sd_v  (N x 1 vec): st. deviations 
    
    % OUTPUT:
    % v  (N x mc matrix): MC samples of random (folded Gaussian) voltages

    v = abs(repmat(sd_v,1,mc).*randn(N,mc) + repmat(vi,1,mc));
    
end
% path_plot(9,N,t(1:end-1)+dt/2,u,xi)
%_______________________________________________________________________
function path_plot(osc,N,tt,u,xi)

    % All mc trials
    figure(1)
    subplot(2,2,1)
    plot(tt,squeeze(u(osc,:,:))); title('MC Speed')
    xlabel('t')
    set(gca,'linewidth',1.5, 'fontsize',20); 
    subplot(2,2,2)
    plot(tt,squeeze(u(osc+N,:,:))); title('MC Angle')
    xlabel('t')
    set(gca,'linewidth',1.5, 'fontsize',20);  
    subplot(2,2,4)
    plot(tt,squeeze(xi(osc,:,:))); title('OU Noise')
    xlabel('t')
    set(gca,'linewidth',1.5, 'fontsize',20); 
end
