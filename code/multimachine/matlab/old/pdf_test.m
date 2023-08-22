
clear; clear; close all
rng('default')       % For reproducibility

load('MC_case30_uncorr_68fail.mat')
% ['MC_case30_uncorr.mat','mc','mc_vec','err_L22',...
%     't','grids','cells','Rho','th','sig','fmax','fmax_time',...
%     'conv_study_time']

N = 30;  % number of buses
linefail = true;  % line fail? true/false

% Number of model and noise equations per oscillator/machine
neq = 2;             % number of state eqns (e.g., speed, angle)
nxi  = 1;            %    nxi = 1 --> Only speeds have noise
                     %    nxi = 2 --> speeds and angles have noise
mc_base = mc;        % number of trials for benchmark mc solution
tmc = t;             % time grid for benchmark MC
ntmc = length(tmc);        % number of time nodes for mc solutions
dtmc = tmc(end)/(ntmc-1);  % time step for mc solutions

tf = tmc(end);      % final time for SDE simulation and PDE
dt = 0.001;         % SDE & PDE time step 
t = (0:dt:tf)';     % PDE time grid 
nt = length(t); 

ratio_t = dtmc/dt;  % ratio of timesteps

%--------------------------------------------------------------------------
% Check inputs
if mod(ratio_t,1)~=0 || ratio_t<1
% state = 17; 
%     fbase = squeeze(fmax(state,1:nx,:)); 
    error('Check t compared to tmc')
end
if t(1) ~= tmc(1) 
    error('Check t compared to tmc')
end
if neq*N ~= length(cells)
    error('total number of states does not match')
end
if neq<=1 || neq>2 || nxi<0 || nxi>neq
    error('Check model constants')
end

%--------------------------------------------------------------------------
% Make sure Matpower files added to path 
matpower_addpath_commands

% choose case from matpower7.1/data
if N==9
    [branch,Pm,Ybus,vi,d0,success] = runpf_edited('case9.m');
    
    if linefail
        % Kill line 8-9
        Ybus(8,9) = 0; Ybus(9,8) = 0;
    end
        
elseif N==30
    [branch,Pm,Ybus,vi,d0,success] = runpf_edited('case30.m');
    
    if linefail
        % Kill line 6-8
        Ybus(8,6) = 0; Ybus(6,8) = 0;
    end
else
    error('runpf_edited only set up for N=9 or 30')
end

if success ~= 1
    error('runpf_edited did not run successfully')
end

% MatPower outputs values for the system in equilibrium:
% dist_mat = (_x4) matrix of branch data
% Pm = vector of total bus power injections
% Ybus = complex admittance matrix
% vi = voltage magnitudes
% d0 = initial angles delta(0) (radians)

% admittance matrices
g = real(Ybus); % for cosines (generators)
b = imag(Ybus); % for sines (buses)

% ODE model parameters
N_ones = ones(N,1);   % Used throughout
H = N_ones;           % Inertia coefficients
D = N_ones;           % Damping coefficients
wr = N_ones;          % Equilibrium speed

% Percentage of equilibriums for st. devs. 
sd_ratio = 0.05;   % voltage component  (5%)

% Choose with method for coefficient learning
%   = "llr" for Loc. Lin. (Gaussian) kernel smoothing regress. w/ 10-fold CV
%   = "lin" for standard ordinary least squares linear regression
method = "lin";

if ~(method=="llr" || method=="lin")
    error('Expecting method = llr or lin')
end

%--------------------------------------------------------------------------
% Correlated OU Noise Parameters and ICs: 

% Correlation Matrix Rho loaded in
% OU noise amplitudes sig loaded in 
% OU drift th loaded in 

C = chol(Rho,'lower');

% Volatility Matrix
vol = repmat(sqrt(2*th).*sig,1,nxi*N).*C;

% Covariance of stationary OU  = cov*cov'
cov = chol(sylvester(diag(th),diag(th),vol*vol'),'lower');

vol = sparse(vol);

% Parameters for random vi (folded gaussian) --> (N x mc matrix)
sd_v = abs(vi)*sd_ratio;         % st. dev--> percent of equilibrium
sd_v(vi==0) = abs(mean(vi))*sd_ratio;

%--------------------------------------------------------------------------                                       
nmc = 4;
mc0 = 500;
mc = nmc*mc0;

u = zeros(neq*N,nt,nmc*mc0);
xi = zeros(nxi*N,nt,nmc*mc0);
v_t = zeros(N,nmc*mc0);
% compute the cost if SDE samples for the different MC trials
tic  % Start first wall clock
for i0 = 1:nmc  
    
    % Temp Stationary ICs for noise
    xi00 = cov*randn(nxi*N,mc0);
    xi0 = cov*randn(nxi*N,mc0);
    
    % temp random voltages
    v_t = get_v(N,mc0,vi,sd_v);
    v(:,(i0-1)*mc0+1:mc0*i0) = v_t;

    % deterministic ICs for states
    u00 = [wr; d0];  u00 = repmat(u00,1,mc0);

    [u_eq, ~] = classical_OU2(mc0,nt,dt,u00,N,H,D,wr,Pm,g,b,v_t,nxi,xi00,th,vol);

    % Compute MC with random ICs
    % We want ICs to be previous data at last time (i.e., equilibrium) 
    u0 = squeeze(u_eq(:,end,:));

    % MC paths of states (u) and noise (xi)
    % Outputs at times t
    [u_t, xi_t] = classical_OU2(mc0,nt,dt,u0,N,H,D,wr,Pm,g,b,v_t,nxi,xi0,th,vol);
    
    u(:,:,(i0-1)*mc0+1:mc0*i0) = u_t; xi(:,:,(i0-1)*mc0+1:mc0*i0) = xi_t;
    disp('IC iteration '); disp(i0); disp(' out of '); disp(nmc)
end
ICMC = toc;
disp('IC Time: '); disp(ICMC)

% plot paths for single machine
% path_plot(1,N,t,u,xi)

%% Advection coeffs and solving PDE

ng = 2; % Number of ghost cells per boundary !!!!! DO NOT CHANGE !!!!!!
fpdes = zeros(neq*N,max(cells)+2*ng,ntmc);
PDEclocks0 = zeros(neq*N,1);
pde_err = zeros(neq*N,1);
mc = 1000;
for k = 33%:neq*N
    disp(k)
    
    % spatial grid
    nx = cells(k);
    xpts = grids(1:nx,k);
    dx = xpts(2) - xpts(1);
    xpts_e = [xpts - dx/2; xpts(end) + dx/2];   % left cell edges
    f_ind = ng+1:nx+ng;     % indices of non-ghost cells for pde
   
    % Benchmark Solution
    fbase = squeeze(fmax(k,1:nx,:));
    
    % which oscillator/machine
    osc = mod(k,N);  
    if osc == 0 % correct when divisble by N
        osc = N;   
    end

    % Get associated params H, D, Pm, and wr for current machine
    hh = H(osc); dd = D(osc); pm = Pm(osc); wwr = wr(osc);
        
    tic

    % Allocate
    ftemp = zeros(nx+2*ng,nt);
    ftemp(f_ind,1) = fbase(1:nx,1);
    t0 = 0;

    for k0 = 2:nt
        t0 = t0 + dt;  % time that we want to step to
        
%         if k<=(N/2) && k0<150
%             method = "llr";
%             if mod(k0,10)==0 || k0 ==2
%                 % data for coeff at previous time point
%                 u_n = squeeze(u(:,k0-1,1:mc)); xi_n = squeeze(xi(:,k0-1,1:mc)); 
%                 [coeff, coemax] = get_coeff(k0,method,mc,N,osc,k,u_n,xi_n,nxi,v(:,1:mc),hh,dd,pm,wwr,b,g,xpts_e,dx);
%             end
%         else
%             method = "lin";
            % data for coeff at previous time point
            u_n = squeeze(u(:,k0-1,1:mc)); xi_n = squeeze(xi(:,k0-1,1:mc)); 
            [coeff, coemax] = get_coeff(k0,method,mc,N,osc,k,u_n,xi_n,nxi,v(:,1:mc),hh,dd,pm,wwr,b,g,xpts_e,dx);
%         end
%         if dt*coemax/dx > 1.1
%             error('CFL > 1')
%         end

        % solution after time step of length dt
        ftemp(f_ind,k0) = lax_wen(ftemp(:,k0-1),f_ind,nx,coeff,dx,dt); 

%             if max(ftemp(:,k0)) > 5*max(fbase(:,1))
%                error('PDE blows up')
%             end
%             if max(isnan(ftemp(f_ind,k0)))==1
%                error('PDE has NaN values')
%             end
            if mod(k0,100)==0
               figure(3)
               plot(xpts,ftemp(f_ind,k0))
               drawnow
               aa=1;
            end
    end
    PDEclocks0(k) = toc;
%     PDEclocks(k) = PDEclocks0(k) + ICMCclocks(count);
    disp('PDE Time: '); disp('PDEclocks0(k)')

    ftemp = ftemp(f_ind,1:ratio_t:end);
    err_temp = sqrt(trapz(tmc,trapz(xpts,(ftemp-fbase).^2))...
                ./trapz(tmc,trapz(xpts,fbase.^2)));
            disp('Error: '); disp(err_temp);

    fpdes(k,1:nx,:) = ftemp;
    pde_err(k) = err_temp; 
end



% % Wall time if we only generate the maximum number of samples needed for
% %   all PDEs at 5% tolerance
% total_pde_time_max = sum(PDEclocks0) + ICMCclocks(count_max);
% 
% % Wall time if we generate new samples for each PDE at 5% tolerance
% total_pde_time = sum(PDEclocks);
% 
% disp('Total PDE Time Max: '); disp(total_pde_time_max)
% disp('Total PDE Time: '); disp(total_pde_time)

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
