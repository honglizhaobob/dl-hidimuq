% MC convergence study

clear; clear; close all
rng('default')       % For reproducibility
tic                  % Start first wall clock

%------------------------------------------------------------------------
mc = 25000;   % Maximum possible MC trials   
dmc = 500;   % Test at increements of dmc 
              % make sure dmc divides mc
mc_vec = (dmc:dmc:10000)';
nmc = length(mc_vec);

tf = 10;            % final time for SDE simulation and PDE
dt = 0.01;          % SDE time step 
t = (0:dt:tf)';     % time grid for PDFs/SDE
nt = length(t); 

%--------------------------------------------------------------------------
% Make sure Matpower files added to path 
matpower_addpath_commands

% choose case from matpower7.1/data
[branch,Pm,Ybus,vi,d0,success] = runpf_edited('case30.m');

if success ~= 1
    error('runpf_edited did not run successfully')
end

% MatPower outputs values for the system in equilibrium:
% dist_mat = (_x4) matrix of branch data
% Pm = vector of total bus power injections
% Ybus = complex admittance matrix
% vi = voltage magnitudes
% d0 = initial angles delta(0) (radians)

% Kill line 6-8
Ybus(10,22) = 0; Ybus(22,10) = 0;

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

%_____________________________________________________________________
% Check inputs
if neq<=1 || neq>2 || nxi<0 || nxi>neq
    error('Check model constants')
end
if dmc<1 || mod(mc,dmc)~=0 
    error('Check mc, dmc, nmc')
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

    % br_b can't = 0 
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

vol = sparse(vol);

%--------------------------------------------------------------------------
% Get data to compute random ICs via MC and KDE:

% MC realizations of vi (folded gaussian) --> (N x mc matrix)
sd_v = abs(vi)*sd_ratio(end);    % st. dev--> percent of equilibrium
sd_v(vi==0) = abs(mean(vi))*sd_ratio(end);
v = get_v(N,mc,vi,sd_v);         % size = n x mc0

% deterministic ICs for states
u00 = [wr; d0];  u00 = repmat(u00,1,mc);

% Stationary ICs for noise
xi00 = cov*randn(nxi*N,mc);

[u_eq, ~] = classical_OU2(mc,nt,dt,u00,N,H,D,wr,Pm,g,b,v,nxi,xi00,th,vol);

clockIC1 = toc; disp('IC Clock 1: '); disp(clockIC1)

%_____________________________________________________________________
% Compute MC with random ICs
tic

% We want ICs to be previous data at last time (i.e., equilibrium) 
u0 = squeeze(u_eq(:,end,:));

% Stationary ICs for noise
xi0 = cov*randn(nxi*N,mc);

% MC paths of states (u) and noise (xi)
% Outputs at times t
[u, ~] = classical_OU2(mc,nt,dt,u0,N,H,D,wr,Pm,g,b,v,nxi,xi0,th,vol);

clockIC2 = toc; disp('IC Clock 2: '); disp(clockIC2)

%% Compute domain for PDFs based on MC paths
tic

% store left and right boundaries and number of cells
bound_a = zeros(neq*N,1); bound_b = zeros(neq*N,1); 
cells = zeros(neq*N,1);

for ist = 1:neq*N

    u_st = squeeze(u(ist,:,:));  % paths for state of interest

    % PDE grid size via Silverman's rule
    sdev = std(u_st,0,2);   % sample st. devs. of paths
    bw_temp = 0.9*min(sdev,iqr(u_st,2)/1.34)*mc^(-0.2);
    dx = min(bw_temp);  % choose smallest over all times

    % Padding PDE domain
    eta = max(sdev);
    % PDE Compuatational boundaries
    bound_a(ist) = min(u_st,[],'all') - eta; 
    bound_b(ist) = max(u_st,[],'all') + eta;

    b_length = bound_b(ist) - bound_a(ist); 
    if b_length <= 0 || isnan(b_length) || abs(b_length) == Inf
       error('b_length <= 0, or = NaN, Inf')
    end
    
    if dx >= b_length || dx == 0 || abs(dx) == Inf || isnan(dx)
        error('dx >= b_length, or = 0, NaN, Inf')
    end 

    % number of cells
    cells(ist) = ceil(b_length/dx); 
    if cells(ist) <= 0 || isnan(cells(ist)) || abs(cells(ist)) == Inf
        error('cells(i1) <= 0, or = NaN, Inf')
    end

end
cellmax = max(cells);

% Store spatial grids
grids = zeros(cellmax,neq*N);
for ig = 1:neq*N
    
    grid_temp = zeros(cellmax,1);
    
    a0 = bound_a(ig); b0 = bound_b(ig);
    dx = (b0-a0)/cells(ig);
    
    grid_temp(1:cells(ig)) = linspace(a0+dx/2,b0-dx/2,cells(ig))';
    grid_temp(cells(ig)+1:end) = NaN;
    grids(:,ig) = grid_temp;
end

clock_dom = toc; disp('Domain Clock: '); disp(clock_dom)

%_____________________________________________________________________
%% MC PDF via Gaussian KDE with maximum number of trials
tic

fmax = zeros(neq*N,cellmax,nt);
L22 = zeros(neq*N,1);      % L2 norm in space and time

for imax = 1:neq*N
    disp(imax)
    
    nx = cells(imax);
    xpts = grids(1:nx,imax);    % spatial grid
    
    for n1 = 1:nt

        u_st_n = squeeze(u(imax,n1,:));
        bw = 0.9*min(std(u_st_n),iqr(u_st_n)/1.34)*(mc^(-0.2));
        
        fmax(imax,1:nx,n1) = ksdensity(u_st_n,xpts,'bandwidth',bw);
         
    end
    L22(imax) = sqrt(trapz(t,trapz(xpts,squeeze(fmax(imax,1:nx,:)).^2)));
end
clock_fmax = toc;

%________________________________________________________
%% Intermediate MC via Gaussian KDE
tic

% get new independent MC trials
v = get_v(N,mc,vi,sd_v);   
% Stationary ICs for noise
xi00 = cov*randn(nxi*N,mc);

[u_eq, ~] = classical_OU2(mc,nt,dt,u00,N,H,D,wr,Pm,g,b,v,nxi,xi00,th,vol);

clockIC3 = toc; disp('IC Clock 3: '); disp(clockIC3)
tic

% We want ICs to be previous data at last time (i.e., equilibrium) 
u0 = squeeze(u_eq(:,end,:));
% Stationary ICs for noise
xi0 = cov*randn(nxi*N,mc);

% MC paths of states (u) and noise (xi)
% Outputs at times t
[u, ~] = classical_OU2(mc,nt,dt,u0,N,H,D,wr,Pm,g,b,v,nxi,xi0,th,vol);

clockIC4 = toc; disp('IC Clock 4: '); disp(clockIC4)

%________________________________________________________
%% Compute errors

tic
err_L22 = zeros(nmc,neq*N);  % L2 error in space and time
kk = 0;                      % initialize counter
max_err = 1;                 % initialize tolerance test

% Find mc trials that give close to 5% rel error
while max_err > 0.05 && kk <= nmc
    
    kk = kk + 1;
    mc0 = mc_vec(kk);
        
    for ii = 1:neq*N

        disp([kk,ii])

        nx = cells(ii);
        xpts = grids(1:nx,ii);   % spatial grid
        fmax_test = squeeze(fmax(ii,1:nx,:));
    
        u_st_mc0 = squeeze(u(ii,:,1:mc0));
        err_temp = zeros(nt,1);

        for nn = 1:nt
            
            u_st_mc0_n = u_st_mc0(nn,:)';
            bw = 0.9*min(std(u_st_mc0_n),iqr(u_st_mc0_n)/1.34)*(mc0^(-0.2));
            ftest = ksdensity(u_st_mc0_n,xpts,'bandwidth',bw); 

            err_temp(nn) = trapz(xpts,(ftest-fmax_test(:,nn)).^2);
        end
        err_L22(kk,ii) = sqrt(trapz(t,err_temp)...
                        /trapz(t,trapz(xpts,squeeze(fmax(ii,1:nx,:)).^2)));        
    end
    
    max_err = max(err_L22(kk,:));
end

err_L22 = err_L22(1:kk,:);
mc_vec = mc_vec(1:kk);

clock_mc = toc; disp('MC Clock: '); disp(clock_mc)

conv_study_time = clock_mc + clockIC3 + clockIC4 + clock_fmax...
                        + clock_dom + clockIC1 + clockIC2;
                    
fmax_time = clock_fmax + clock_dom + clockIC1 + clockIC2;               

disp('Total CPU Time: '); disp(conv_study_time)

figure(2)
semilogy(mc_vec,err_L22(:,1:N))
title('L^2 Rel. Error for speeds')
xlabel('MC trials')

figure(3)
semilogy(mc_vec,err_L22(:,N+1:neq*N))
title('L^2 Rel. Error for angles')
xlabel('MC trials')

save('MC_case30_uncorr_10_22fail.mat','mc','mc_vec','err_L22',...
    't','grids','cells','Rho','th','sig','fmax','fmax_time',...
    'conv_study_time','-v7.3')
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