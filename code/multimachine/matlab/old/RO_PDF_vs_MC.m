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

%     LAST UPDATE: Nov. 9, 2021
              
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
%                 RO-OU-no-gov
%                 
%     NEEDED FILES:
%        matpower_addpath_commands.m (WARNING: depends on local path)
%        runpf_edited.m
%        classical_OU.m
%        get_coeff.m
%        regress_ll.m
%        ksrlin.m
%        lax_wen.m


%     Simplified NON-ADAPTIVE VERSION:
%        Generates SDE samples ahead of time outside time loop for PDE.
%        ICs are also found via MC
%--------------------------------------------------------------------------

%  clear; clear; close all
rng('default')       % For reproducibility

load('MC_case30_con_68fail.mat')
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
dt = 0.005;         % SDE & PDE time step 
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
% Find how many trials MC needs for each state
mc_ind = zeros(neq*N,1); 
state_samples_mc = mc_ind;
for i = 1:neq*N
    [err_temp, shift] = sort(err_L22(:,i));
    mc_ind(i) = shift(find(err_temp<=0.05, 1, 'last' ));
    state_samples_mc(i) = mc_vec(mc_ind(i));
end
% speed = sum(state_samples_mc(1:N))
% phase = sum(state_samples_mc((N+1):end))
% return

nmc = length(mc_vec);
ICMCclocks = zeros(nmc,1);   % store wall clocks for copmuting trials

mc0 = 500;
u = zeros(neq*N,nt,mc_vec(end));
xi = zeros(nxi*N,nt,mc_vec(end));
v_t = zeros(N,mc_vec(end));
% compute the cost if SDE samples for the different MC trials
for i0 = 1:nmc   
    
    tic  % Start first wall clock
    
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

    if i0 == 1
        ICMCclocks(i0) = toc; 
    else
        ICMCclocks(i0) = ICMCclocks(i0-1) + toc; 
    end
    disp('IC iteration '); disp(i0); disp(' out of '); disp(nmc)
    disp('IC Time: '); disp(ICMCclocks(i0))
end

% plot paths for single machine
% path_plot(1,N,t,u,xi)

%% MC solutions with for different number of trials

% fmc = zeros(neq*N,max(cells),nt);
% MCclocks = zeros(neq*N,1);
% MCclocks0 = MCclocks;
% mc_err = zeros(neq*N,1);
% 
% for j = 1:neq*N
%     
%     disp('MC state iteration: '); disp(j); disp(' out of '); disp(neq*N);
%     tic
%     
%     % spatial grid
%     nx = cells(j);
%     xpts = grids(1:nx,j);
%     
%     % number of trials for this state
%     mc = mc_vec(mc_ind(j));
%     
%     for j0 = 1:nt
%     
%         u_st = squeeze(u(j,j0,1:mc));  % mc data for state of interest
% 
%         % KDE with Silverman's rule
%         bw = 0.9*min(std(u_st),iqr(u_st)/1.34)*mc^(-0.2);
%         fmc(j,1:nx,j0) = ksdensity(u_st,xpts,'bandwidth',bw);
%     end
%     
%     % Time for MC trials + time for KDE at each state
%     MCclocks0(j) = toc;
%     MCclocks(j) = MCclocks0(j) + ICMCclocks(mc_ind(j));
%     mc_err(j) = err_L22(mc_ind(j),j);
% end
% 
% % maximum number of samples used over all marginals
% mc_mc_max = mc_vec(end);
% 
% % Wall time if we only generate the maximum number of samples needed for
% %   all marginals at 5% tolerance
% total_mc_time_max = sum(MCclocks0) + ICMCclocks(end);
% 
% % Wall time if we generate new samples for each marginal at 5% tolerance
% total_mc_time = sum(MCclocks);
% % Adjust fmc times to match fmax times
% fmc = fmc(:,:,1:ratio_t:end);
% 
% disp('Total MC Time Max: '); disp(total_mc_time_max)
% disp('Total MC Time: '); disp(total_mc_time)

%% Advection coeffs and solving PDE

ng = 2; % Number of ghost cells per boundary !!!!! DO NOT CHANGE !!!!!!
fpdes = zeros(neq*N,max(cells)+2*ng,ntmc);
PDEclocks = zeros(neq*N,1);
PDEclocks0 = PDEclocks;
pde_err = zeros(neq*N,1);
sample_states_pde = zeros(neq*N,1);
count_max = 0;

factor = 1;
dt_new = dt/factor;
nt_new = (nt-1)*factor + 1;

for k = 1:30%31:neq*N
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
    
    % Initialize
    err_temp = 0;
    err = 100;
    count = 0;
    
    if k<=N
        nmc = 20;%state_samples_mc(k)/250 ;
    else
        nmc = state_samples_mc(k)/100;
    end
    
    while err_temp < err && count <= nmc
        
        tic
        count = count + 1; disp(count);
        count_max = max(count_max,count);
%         mc = mc_vec(count);
        if k<=N
            mc = count*250;
        else
            mc = count*100;
        end
        
        % Allocate
        ftemp = zeros(nx+2*ng,nt_new);
        ftemp(f_ind,1) = fbase(1:nx,1);
        t0 = 0;
        
            
        disp([k,count])

        for k0 = 2:nt_new
            t0 = t0 + dt_new;  % time that we want to step to
            
%             if k0<650
                method="lin";
%             else
%                 method="lin";
%             end

            % data for coeff at previous time point
            if mod(k0-1,factor)==0 || k0 == 2
    
            k0_n = (k0-1)/factor + 1;
            if k0 == 2
                k0_n = 2;
            end

            u_n = squeeze(u(:,k0_n-1,1:mc)); xi_n = squeeze(xi(:,k0_n-1,1:mc)); 

            % Calculate advection coefficient at for next time
            %    defined on xpts_e (size(coeff) = size(xpts_e)) 
            [coeff, coemax] = get_coeff(k0_n,method,mc,N,osc,k,u_n,xi_n,nxi,v(:,1:mc),hh,dd,pm,wwr,b,g,xpts_e,dx);
%             if dt_new*coemax/dx > 1
%                 error('CFL > 1')
%             end
            end

            % solution after time step of length dt
            ftemp(f_ind,k0) = lax_wen(ftemp(:,k0-1),f_ind,nx,coeff,dx,dt_new); 

%             if max(ftemp(:,k0)) > 5*max(fbase(:,1))
%                error('PDE blows up')
%             end
%             if max(isnan(ftemp(f_ind,k0)))==1
%                error('PDE has NaN values')
%             end
%             if mod(k0,10)==0
%                figure(3)
%                plot(xpts,ftemp(f_ind,k0))
%                drawnow
%             end

        end
        PDEclocks0(k) = toc;
%         PDEclocks(k) = PDEclocks0(k) + ICMCclocks(count);
        if count>1
            err = err_temp;
        end

        ftemp = ftemp(f_ind,1:(factor*ratio_t):end);
        err_temp = sqrt(trapz(tmc,trapz(xpts,(ftemp-fbase).^2))...
                    ./trapz(tmc,trapz(xpts,fbase.^2)));
               
    end
    sample_states_pde(k) = mc - 250;

    fpdes(k,1:nx,:) = ftemp;
    pde_err(k) = err_temp;
end
sum(sample_states_pde(1:N))
sum(sample_states_pde((N+1):end))

% maximum number of samples used over all PDEs
pde_mc_max = mc_vec(count_max);  

% % Wall time if we only generate the maximum number of samples needed for
% %   all PDEs at 5% tolerance
% total_pde_time_max = sum(PDEclocks0) + ICMCclocks(count_max);
% 
% % Wall time if we generate new samples for each PDE at 5% tolerance
% total_pde_time = sum(PDEclocks);
% 
% disp('Total PDE Time Max: '); disp(total_pde_time_max)
% disp('Total PDE Time: '); disp(total_pde_time)

% save('errors_case30_uncorr_68fail.mat','ICMCclocks','u','xi','total_pde_time',...
%     'total_pde_time_max','PDEclocks','PDEclocks0','MCclocks','MCclocks0',...
%     'total_mc_time','total_mc_time_max','mc_mc_max','pde_mc_max',...
%     'pde_err','mc_err','fpdes','fmc','fmax','grids','cells','t','tmc',...
%     'sample_states_pde','sample_states_mc','-v7.3')

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