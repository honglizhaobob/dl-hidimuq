%     Hongli Zhao, Givens Associate, Argonne Nat. Lab.,
%     CAM PhD Student, UChicago
%     Last updated: Aug 10, 2023
%% Define simulation parameters
clear; rng('default'); startup;
% Make sure Matpower files added to path 
run("./matpower7.1/startup.m");

% choose case from matpower7.1/data
[dist, Pm, amat, vi, d0, success] = runpf_edited('case9.m');
% get real and imaginary parts

% real part
g = real(amat);
% imag part
b = imag(amat);

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

mc = 10000;             % Number of MC paths
tf = 50.0;               % final time for simulation
dt = 0.01;             % learn PDE coefficients in increments of dt
time = 0:dt:tf;        % coarse uniform time grid
tt = time;
nt = length(time);     % number of time steps
% Sanity check: simulate deterministic equations with optimal initial conditions
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

% Burn-in period

% Allocate random initial conditions:
u0 = zeros(mc,4*n);

% Random voltages . (folded gaussian, mean at optimal vi given by matpower)
sd_v = mean(vi)*0.01;
v = abs(sd_v*randn(mc,n) + reshape(vi,1,[]));

% initialize speeds at equilibrium =wr
u0_w = wr*ones(mc,n);

% initial angles at d0
u0_d = repmat(d0',mc,1);


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

% save / load burned initial conditions
fname = "./data/case9_ic.mat";
if ~isfile(fname)
    disp("burning ICs ");
    % burn samples for tf time
    burn_mc = classical_mc(mc,dt,nt,u0,alpha,theta,C,H,D,Pm,wr,g,b);
    % save
    disp("saving... ");
    burned_ic = burn_mc(:,:,end);
    save(fname,"burned_ic","-v7.3");
else
    disp("IC computed, loading...");
    tmp = load(fname);
    burned_ic = tmp.burned_ic;
end

u0 = burned_ic;
% simulate paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Case 9: RO-PDF problem set up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tf = 10.0;
dt = 0.01;           % learn PDE coefficients in increments of dt
tt = 0:dt:tf;        % coarse uniform time grid
nt = length(tt);     % number of time steps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Line removal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
remove_line = true;
if remove_line
    % remove (8, 9), see: Page 70 of https://link.springer.com/chapter/10.1007/BFb0044331
    % for network topology. And measure (4, 9), (8, 7), adjacent lines.
    trip_from = 8; trip_to = 9;
    % assign admittance to 0
    g(trip_from,trip_to)=0.0; 
    b(trip_from,trip_to)=0.0;
end
%%
fname = "./data/case9_mc_data.mat";
if ~isfile(fname)
    disp("simulating trajectories ...");
    % simulate SDE with burned IC
    paths_mc = classical_mc(mc,dt,nt,u0,alpha,theta,C,H,D,Pm,wr,g,b);
    % save
    save(fname,"-v7.3");
else
    disp("loading simulated trajectories ...");
    load(fname);
end

%% Visualize trajectories
close all
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


%% Compute energy for a specific line given paths data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SELECT TWO LINES
from_line1 = 4;
to_line1 = 9;
from_line2 = 7;
to_line2 = 8;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% save / load
fname = "./data/case9_mc1d_coeff_data.mat";
if ~isfile(fname)
    disp("running conditional expectation from scratch. ")
    % compute energy for each Monte Carlo trial at each time point
    mc_energy1 = zeros(mc,nt);
    mc_energy2 = zeros(mc,nt);
    % compute expectation output for each Monte Carlo trial at each time point
    mc_condexp_target1 = zeros(mc,nt);
    mc_condexp_target2 = zeros(mc,nt);
    for i =1:mc
        disp(i)
        for j = 1:nt
            % get solution
            u_i = reshape(paths_mc(i,:,j), [], 1);
            mc_energy1(i,j)=line_energy(b,from_line1,to_line1,u_i);
            mc_condexp_target1(i,j)=condexp_target(b,from_line1,to_line1,u_i);
            mc_energy2(i,j)=line_energy(b,from_line2,to_line2,u_i);
            mc_condexp_target2(i,j)=condexp_target(b,from_line2,to_line2,u_i);
        end
    end
    disp("finished ... saving. ")
    save(fname, "tt", "mc_energy1", "mc_condexp_target1", ...
        "mc_energy2", "mc_condexp_target2", "-v7.3");
else
    load(fname);
end

%% Learn coffcients & solve 1d marginal PDE via lax-wendroff (Line 4-9)
close all; 
rng('default');

% line is fixed and data simulated above (1d)
from_line=from_line1; to_line=to_line1; 
mc_energy=mc_energy1; 
mc_condexp_target=mc_condexp_target1;
% set up pde domain
dx = 0.02;          % spatial step size
ng = 2;             % number of ghost cells
a0 = 0.0;            % energy cannot be negative
b0 = max(mc_energy,[],"all"); % padded right boundary
nx = ceil((b0-a0)/dx);                      % number of nodes
xpts = linspace(a0+0.5*dx, b0-0.5*dx,nx)';  % column vector of cell centers
dx = xpts(2)-xpts(1);                       % updated dx if needed
xpts_e = [xpts-0.5*dx; xpts(end)+0.5*dx];   % cell edges for advection coeff

% allocate pde solution (IC from kernel density estimate)
f = zeros(nx+2*ng,nt);
f_ind = ng+1:nx+ng;     % indices of non-ghost cells

f0 = [squeeze(mc_energy(:,1))];
bw = 0.9*min(std(f0), iqr(f0)/1.34)*(mc)^(-0.2);
f(f_ind,1) = ksdensity(f0,xpts,'bandwidth',bw);
figure(1);
plot(xpts, f(f_ind,1),"LineWidth",1.5,"Color","black");

% failure level for energy, determined heuristically
% both picked lines have rateA 2.5 p.u., we choose 1.0 p.u., 40% of long term rating
failure_level = 1.00;

% Find the temporal index of peak operating line energy, we only visualize
% that time step
max_energy1 = max(max(mc_energy1));
[~,max_energy1_time_idx] = find(mc_energy1==max_energy1);
%% Begin RO-PDF

% reduced order MC trial numbers
mcro = 5000;
% time loop
for nn = 2:nt
    % current time of simulation
    curr_time = dt*nn;
    disp(nn)
    % learn advection coeffcient via data and propagate PDE dynamics
    % Exact solution is of form: E[Y | X]

    % get X data (previous time)
    energy_data = squeeze(mc_energy(1:mcro,nn-1));
    
    % get Y data (previous time)
    response_data = squeeze(mc_condexp_target(1:mcro,nn-1));

    % compute advection coefficient (need to be defined on cell centers)

    % Get adv. coefficient defined on xpts_e (size(coeff) = size(xpts_e))
    coeff = get_coeff(energy_data,response_data,xpts_e,"lin");

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
        end   
        f(f_ind,nn) = f_temp(f_ind);
        
    end    
    
    f_pred = f(f_ind,nn);
    if max(abs(f(:,nn)))>1e2
        error('PDE blows up')
    end
    if max(isnan(f(:,nn)))==1
        error('PDE has NaN values')
    end

    % visualize learned coeffs, RO-PDF solution and exact solution from KDE
    if mod(nn,1)==0
        %%
        % KDE ground truth
        f0 = [squeeze(mc_energy(:,nn))];
        bw = 0.9*min(std(f0), iqr(f0)/1.34)*(mc)^(-0.2);
        f_kde = ksdensity(f0,xpts,'Support','positive');

        % ensure density
        f_kde = f_kde/trapz(dx,f_kde);
        f_pred = f_pred/trapz(dx,f_pred);

        % save at maximum operating time
        fig = figure(2); fig.Position = [500 800 800 500];
        % save PDF
        plot(xpts,f_pred,"-.","LineWidth",5.0,"Color","#7E2F8E"); hold on;
        plot(xpts,f_kde,"-","LineWidth",5.0,"Color",[0 0 0 0.8]); hold off;
        title(sprintf("$t=%.2f$",curr_time), ...
            "FontSize",24,"FontName","Times New Roman", ...
            "Interpreter","latex",'fontweight','bold');
        % axes labels
        xlabel("Line 4-9 Energy","FontSize",30,"FontName","Times New Roman");
        ylabel("PDF","FontSize",30,"FontName","Times New Roman");
        ax = gca;
        ax.FontSize = 30; box on;
        ax = gca; ax.LineWidth = 2;
        % legend for lines
        legend(["Pred","KDE"],"FontSize",30,"Location","northeast");

        
        % save large plot
        if nn == max_energy1_time_idx
            figure_name = "./fig/case9/line1/max_energy1_time_PDF.png";
            % save figure
            exportgraphics(gcf,figure_name,"Resolution",300);
        end

        % plot Zoomed in plot
        fig = figure(3); fig.Position = [500 800 500 500];
        % find positions > 0.95
        idx = find(xpts>0.95,1);
        plot(xpts(idx:end),f_pred(idx:end),"-.","LineWidth",6.0,"Color","#7E2F8E"); hold on;
        plot(xpts(idx:end),f_kde(idx:end),"-","LineWidth",6.0,"Color","black"); hold off;
        ax = gca;
        ax.FontSize = 48; box on;
        ax = gca; ax.LineWidth = 2;
        
        % change units for y axis
        ax = gca; 
        ax.YAxis.Exponent = -2;

        % plot threshold
        xl = xline(failure_level,'-.', ...
            "Threshold","LineWidth",4.0, ...
            "Color","red", ...
            'DisplayName','Threshold');
        xl.LabelVerticalAlignment = 'middle';
        xl.LabelHorizontalAlignment = 'center';
        hold on;
        % shade area under KDE curve
        idx = find(xpts>1.0,1);
        ar = area(xpts(idx:end),f_kde(idx:end),"FaceColor","red");
        ar.FaceAlpha = 0.4;
        hold off;

        % save small plot
        if nn == max_energy1_time_idx
            figure_name = "./fig/case9/line1/max_energy1_time_PDF_small.png";
            % save figure
            exportgraphics(gcf,figure_name,"Resolution",300);
        end

        %%

        % calculate probabilities and save to .mat
        if nn == max_energy1_time_idx
            idx = find(xpts>1.0,1);

            % numerically integrate KDE
            %tail_pred = 1-cumtrapz(dx,f_pred);
            %tail_kde = 1-cumtrapz(dx,f_kde);
            %tail_pred = tail_pred(idx);
            %tail_kde = tail_kde(idx);

            % use empirical CDF/survivor (11/09/2023)
            tail_pred = 1-cumtrapz(dx,f_pred);
            tail_kde = 1-empirical_cdf(f0,xpts);
            tail_pred = tail_pred(idx);
            tail_kde = tail_kde(idx);

            disp(tail_pred)
            disp(tail_kde)
            T = table(tail_pred, tail_kde, 'VariableNames', ...
                { 'tail_pred', 'tail_kde'} );
            % Write data to text file
            writetable(T, "./fig/case9/line1/exceedance.txt");
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % CDF plot
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % save at t = 1.0, 2.0, 4.0, 8.0
%         if curr_time == 1.0 || curr_time == 2.0 || ... 
%                 curr_time == 4.0 || curr_time == 8.0
%             figure_name = strcat("./fig/CASE9_ROPDF_CDF_Time_", ...
%                 num2str(curr_time),".png");
%             % Save figures for reporting 
%             fig=figure(2);
%             fig.Position = [500 500 500 500];
%             % estimate CDF
%             F_pred = cumtrapz(f_pred*dx);
%             F_kde = cumtrapz(f_kde*dx);
%             plot(xpts,F_pred,"LineWidth",3.0);
%             hold on;
%             plot(xpts,F_kde,"--","LineWidth",5.0,"Color",[0 0 0 0.5]);
%     
%             title("Case 9","FontSize",18,"FontName","Times New Roman");
%             xlabel("Line 4-9 Energy","FontSize",18,"FontName","Times New Roman");
%             ylabel("CDF","FontSize",18,"FontName","Times New Roman");
%             ax = gca;
%             ax.FontSize = 18;
%             box on;
%             ax = gca;
%             ax.LineWidth = 2;
            % add more lines 
%             hold on;
%             if curr_time == 8.0
%                 legend(["t = 1.0", "", "t = 2.0", "", "t = 4.0", ...
%                     "", "t = 8.0", "Benchmark"], ...
%                     "FontSize",16, ...
%                     "Location","southeast");
%                 % plot failure level
%                 xl = xline(failure_level,'-.', ...
%                     "Threshold","LineWidth",3.0, ...
%                     "Color","red", ...
%                     'DisplayName','Threshold');
%                 xl.LabelVerticalAlignment = 'middle';
%                 xl.LabelHorizontalAlignment = 'center';
%                 % save figure
%                 exportgraphics(gcf,figure_name,"Resolution",300);
%                 disp(strcat("Figure saved at t = ",num2str(curr_time)));
%             end
%         end


    end
end
%% Save computed solutions
fname = "./data/case9/line1_ropdf.mat";
if ~isfile(fname)
    save(fname,"f","a0","b0","dx","mcro");
end

%% Compute and save MC (for comparison, 11/08/2023)
mcro = 2^12;
f_kde = zeros(nx+2*ng,nt);
f_ind = ng+1:nx+ng;

for nn = 2:nt
    % current time of simulation
    curr_time = dt*nn;
    disp(nn)
    % learn advection coeffcient via data and propagate PDE dynamics
    % Exact solution is of form: E[Y | X]

    % get X data (previous time)
    energy_data = squeeze(mc_energy(1:mcro,nn-1));
    
    % get Y data (previous time)
    response_data = squeeze(mc_condexp_target(1:mcro,nn-1));
    f0 = [squeeze(mc_energy(:,nn))];
    tmp = ksdensity(f0,xpts,'Support','positive');
    % ensure density
    tmp = tmp/trapz(dx,tmp);
    f_kde(3:end-2, nn) = tmp;
end

%% Save MC solutions (for comparison, 11/08/2023)
fname = "./data/case9/line1_kde.mat";
if ~isfile(fname)
    save(fname,"f_kde","a0","b0","dx","mcro");
end
%% Learn coffcients & solve 1d marginal PDE via lax-wendroff (Line 7-8)
close all; 
rng('default');

% line is fixed and data simulated above (1d)
from_line=from_line2; to_line=to_line2; 
mc_energy=mc_energy2; 
mc_condexp_target=mc_condexp_target2;
% set up pde domain
dx = 0.02;          % spatial step size
ng = 2;             % number of ghost cells
a0 = 0.0;            % energy cannot be negative
b0 = max(mc_energy,[],"all"); % padded right boundary
nx = ceil((b0-a0)/dx);                      % number of nodes
xpts = linspace(a0+0.5*dx, b0-0.5*dx,nx)';  % column vector of cell centers
dx = xpts(2)-xpts(1);                       % updated dx if needed
xpts_e = [xpts-0.5*dx; xpts(end)+0.5*dx];   % cell edges for advection coeff

% allocate pde solution (IC from kernel density estimate)
f = zeros(nx+2*ng,nt);
f_ind = ng+1:nx+ng;     % indices of non-ghost cells

f0 = [squeeze(mc_energy(:,1))];
bw = 0.9*min(std(f0), iqr(f0)/1.34)*(mc)^(-0.2);
f(f_ind,1) = ksdensity(f0,xpts,'bandwidth',bw);
figure(1);
plot(xpts, f(f_ind,1),"LineWidth",1.5,"Color","black");

% failure level for energy, determined heuristically
% both picked lines have rateA 2.5 p.u., we choose 1.0 p.u., 40% of long term rating
failure_level = 1.00;

% Find the temporal index of peak operating line energy, we only visualize
% that time step
max_energy = max(max(mc_energy));
[~,max_energy_time_idx] = find(mc_energy==max_energy);
%% Begin RO-PDF
% L^2 error in space, over time
all_l2_err = [];
% reduced order MC trial numbers
mcro = 5000;
% time loop
for nn = 2:nt
    % current time of simulation
    curr_time = dt*nn;
    disp(nn)
    % learn advection coeffcient via data and propagate PDE dynamics
    % Exact solution is of form: E[Y | X]

    % get X data (previous time)
    energy_data = squeeze(mc_energy(1:mcro,nn-1));
    
    % get Y data (previous time)
    response_data = squeeze(mc_condexp_target(1:mcro,nn-1));

    % compute advection coefficient (need to be defined on cell centers)

    % Get adv. coefficient defined on xpts_e (size(coeff) = size(xpts_e))
    coeff = get_coeff(energy_data,response_data,xpts_e,"lin");

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
        end   
        f(f_ind,nn) = f_temp(f_ind);
        
    end    
    
    f_pred = f(f_ind,nn);
    if max(abs(f(:,nn)))>1e2
        error('PDE blows up')
    end
    if max(isnan(f(:,nn)))==1
        error('PDE has NaN values')
    end

    % visualize learned coeffs, RO-PDF solution and exact solution from KDE
    if mod(nn,1)==0
        %%
        % KDE ground truth
        f0 = [squeeze(mc_energy(:,nn))];
        bw = 0.9*min(std(f0), iqr(f0)/1.34)*(mc)^(-0.2);
        f_kde = ksdensity(f0,xpts,'Support','positive');

        % ensure density
        f_kde = f_kde/trapz(dx,f_kde);

        f_pred = f(f_ind,nn);
        f_pred = f_pred/trapz(dx,f_pred);

        % save at maximum operating time
        fig = figure(2); fig.Position = [500 800 800 500];
        % save PDF
        plot(xpts,f_pred,"-.","LineWidth",5.0,"Color","#7E2F8E"); hold on;
        plot(xpts,f_kde,"-","LineWidth",5.0,"Color",[0 0 0 0.6]); hold off;
        title(sprintf("$t=%.2f$",curr_time), ...
            "FontSize",24,"FontName","Times New Roman", ...
            "Interpreter","latex",'fontweight','bold');
        % axes labels
        xlabel("Line 7-8 Energy","FontSize",30,"FontName","Times New Roman");
        ylabel("PDF","FontSize",30,"FontName","Times New Roman");
        ax = gca;
        ax.FontSize = 30; box on;
        ax = gca; ax.LineWidth = 2;
        % legend for lines
        legend(["Pred","KDE"],"FontSize",30,"Location","northeast");

        
        % save large plot
        if nn == max_energy_time_idx
            figure_name = "./fig/case9/line2/max_energy_time_PDF.png";
            % save figure
            exportgraphics(gcf,figure_name,"Resolution",300);
        end

        % plot Zoomed in plot
        fig = figure(3); fig.Position = [500 800 500 500];
        % find positions > 0.95
        idx = find(xpts>0.95,1);
        plot(xpts(idx:end),f_pred(idx:end),"-.","LineWidth",6.0,"Color","#7E2F8E"); hold on;
        plot(xpts(idx:end),f_kde(idx:end),"-","LineWidth",6.0,"Color",[0 0 0 0.6]); hold off;
        ax = gca;
        ax.FontSize = 48; box on;
        ax = gca; ax.LineWidth = 2;
        % plot threshold
        xl = xline(failure_level,'-.', ...
            "Threshold","LineWidth",4.0, ...
            "Color","red", ...
            'DisplayName','Threshold');
        xl.LabelVerticalAlignment = 'middle';
        xl.LabelHorizontalAlignment = 'center';
        hold on;
        % change units for y axis
        ax = gca; 
        ax.YAxis.Exponent = -2;


        % shade area under KDE curve
        idx = find(xpts>1.0,1);
        ar = area(xpts(idx:end),f_kde(idx:end),"FaceColor","red");
        ar.FaceAlpha = 0.4;
        hold off;

        % save small plot
        if nn == max_energy_time_idx
            figure_name = "./fig/case9/line2/max_energy_time_PDF_small.png";
            % save figure
            exportgraphics(gcf,figure_name,"Resolution",300);
        end
        %%

        % calculate probabilities and save to .mat
        if nn == max_energy_time_idx
            idx = find(xpts>1.0,1);
            %tail_pred = 1-cumtrapz(dx,f_pred);
            %tail_kde = 1-cumtrapz(dx,f_kde);
            %tail_pred = tail_pred(idx);
            %tail_kde = tail_kde(idx);
            
            % use empirical CDF/survivor (11/09/2023)
            tail_pred = 1-cumtrapz(dx,f_pred);
            tail_kde = 1-empirical_cdf(f0,xpts);
            tail_pred = tail_pred(idx);
            tail_kde = tail_kde(idx);
            disp(tail_pred)
            disp(tail_kde)
            T = table(tail_pred, tail_kde, 'VariableNames', ...
                { 'tail_pred', 'tail_kde'} );
            % Write data to text file
            writetable(T, "./fig/case9/line2/exceedance.txt");
        end
    end
end

%% Save computed solutions
fname = "./data/case9/line2_ropdf.mat";
if ~isfile(fname)
    save(fname,"f","a0","b0","dx","mcro");
end


%% Compute and save MC (for comparison, 11/08/2023)
mcro = 2^12;
f_kde = zeros(nx+2*ng,nt);
f_ind = ng+1:nx+ng;

for nn = 2:nt
    % current time of simulation
    curr_time = dt*nn;
    disp(nn)
    % learn advection coeffcient via data and propagate PDE dynamics
    % Exact solution is of form: E[Y | X]

    % get X data (previous time)
    energy_data = squeeze(mc_energy(1:mcro,nn-1));
    
    % get Y data (previous time)
    response_data = squeeze(mc_condexp_target(1:mcro,nn-1));
    f0 = [squeeze(mc_energy(:,nn))];
    tmp = ksdensity(f0,xpts,'Support','positive');
    % ensure density
    tmp = tmp/trapz(dx,tmp);
    f_kde(3:end-2, nn) = tmp;
end

%% Save MC solutions (for comparison, 11/08/2023)
fname = "./data/case9/line2_kde.mat";
if ~isfile(fname)
    save(fname,"f_kde","a0","b0","dx","mcro");
end







