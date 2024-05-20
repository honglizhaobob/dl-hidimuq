%     Hongli Zhao, Givens Associate, Argonne Nat. Lab.,
%     CAM PhD Student, UChicago
%     Last updated: Aug 10, 2023
%% Define simulation parameters
clear; rng('default');
% Make sure Matpower files added to path 
run("./matpower7.1/startup.m");

% choose case from matpower7.1/data
[dist, Pm, amat, vi, d0, success] = runpf_edited('case57.m');
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
case_number = 57;
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
fname = "./data/case57_ic.mat";
if ~isfile(fname)
    disp("burning ICs ");
    % burn samples for tf time for all mc samples
    for i = 1:mc
        i
        for j = 1:nt
            u0(i,:)=classical_mc_step(dt,u0(i,:),alpha,theta,C,H,D,Pm,wr,g,b);
        end
    end
    
    % save
    disp("saving... ");
    burned_ic = u0;
    save(fname,"burned_ic","-v7.3");
else
    disp("IC computed, loading...");
    tmp = load(fname);
    burned_ic = tmp.burned_ic;
end

u0 = burned_ic;

% simulate paths
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Case 57: RO-PDF problem set up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tf = 8.0;
dt = 0.01;           % learn PDE coefficients in increments of dt
tt = 0:dt:tf;        % coarse uniform time grid
nt = length(tt);     % number of time steps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Line removal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
remove_line = true;
if remove_line
    % remove line (36, 37), consider the energy of adjacent lines (36, 40)
    % and (35, 36)
    trip_from = 36; trip_to = 37;
    % assign admittance to 0
    g(trip_from,trip_to)=0.0; 
    b(trip_from,trip_to)=0.0;
end

% adjacent lines to compute energy
from_line1 = 35; to_line1 = 36;
from_line2 = 36; to_line2 = 40;

% preallocate 
mc_energy1 = zeros(mc,nt);
mc_energy2 = zeros(mc,nt);
mc_target1 = zeros(mc,nt);
mc_target2 = zeros(mc,nt);

fname = "./data/case57_mc_data.mat";
if ~isfile(fname)
    disp("simulating trajectories and computing energy ...");
    for i = 1:nt
        i
        for j = 1:mc
            % step SDE
            u0(j,:)=classical_mc_step(dt,u0(j,:),alpha,theta,C,H,D,Pm,wr,g,b);
            % compute energy
            mc_energy1(j,i)=line_energy(b,from_line1,to_line1,u0(j,:));
            mc_energy2(j,i)=line_energy(b,from_line2,to_line2,u0(j,:));
            % compute regression response
            mc_target1(j,i)=condexp_target(b,from_line1,to_line1,u0(j,:),wr);
            mc_target2(j,i)=condexp_target(b,from_line2,to_line2,u0(j,:),wr);
        end
    end
    % save
    disp("finished ... saving. ")
    save(fname, "mc_energy1", "mc_target1", ...
        "mc_energy2", "mc_target2", "-v7.3");
else
    disp("loading simulated trajectories ...");
    load(fname);
end


%% Learn coffcients & solve 1d marginal PDE via lax-wendroff (line 1)
close all; 
rng('default');

% line is fixed and data simulated above
from_line = from_line1; 
to_line=to_line1; 
% adjust factor (Julia estimated: (35,36)=>rateA:16.33 p.u.
adjust_factor = 0.17;
mc_energy=mc_energy1*adjust_factor; 
mc_condexp_target=mc_target1*adjust_factor;
% set up pde domain
dx = 0.01;          % spatial step size
ng = 2;             % number of ghost cells
a0 = -dx;            % energy cannot be negative
b0 = max(mc_energy,[],"all")+1.0*std(mc_energy,[],"all"); % padded right boundary
nx = ceil((b0-a0)/dx);                      % number of nodes
xpts = linspace(a0+0.5*dx, b0-0.5*dx,nx)';  % column vector of cell centers
dx = xpts(2)-xpts(1);                       % updated dx if needed
xpts_e = [xpts-0.5*dx; xpts(end)+0.5*dx];   % cell edges for advection coeff

% allocate pde solution (IC from kernel density estimate)
f = zeros(nx+2*ng,nt);
f_ind = ng+1:nx+ng;     % indices of non-ghost cells

% start PDE solver at start_idx
start_idx = 200;
f0 = [squeeze(mc_energy(:,start_idx))];
bw = 0.9*min(std(f0), iqr(f0)/1.34)*(mc)^(-0.2);
f(f_ind,start_idx) = ksdensity(f0,xpts,'bandwidth',bw,"Support","positive","BoundaryCorrection","reflection");
figure(1);
plot(xpts, f(f_ind,start_idx),"LineWidth",1.5,"Color","black");

% find max mean operating energy
mean_energy = mean(mc_energy,1)';
max_energy = max(mean_energy);
max_energy_time_idx = find(mean_energy==max_energy); 

% set failure level
failure_level = 16.33;
%% Begin RO-PDF

% L^2 error in space, over time
all_l2_err = [];

% reduced order MC trial numbers
mcro = 5000;
% time loop
for nn = start_idx+1:nt
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
    coeff = get_coeff(energy_data,response_data, ...
        xpts_e,"lin");

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
        % using same coeff., and then output solution on the coarser time
        % grid.
        
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

    % visualize learned coeffs, RO-PDF solution and exact solution from KDE
    fig=figure(1);
    fig.Position = [100 500 1600 400];
    
    % RO-PDF predictions
    set(gca,'linewidth',1.5, 'fontsize',20)
    f_pred = f(f_ind,nn);
    
    plot(xpts,f_pred,'-b','linewidth',2);
    xlabel('x')
    title('Marginal PDF');
    hold on;
    f0 = [squeeze(mc_energy(:,nn))];
    bw = 0.9*min(std(f0), iqr(f0)/1.34)*(mc)^(-0.2);
    f_kde = ksdensity(f0,xpts,'Support','positive', ...
        'BoundaryCorrection','reflection');
    plot(xpts,f_kde,"--", "Color", "black",'linewidth',2);
    hold off;
    
    % compute L^2 error
    tmp  =trapz(dx,(f_kde-f_pred).^2)/trapz(dx,f_kde.^2);
    disp(tmp)

    % ------------------------------------------------------------
    % Save at max mean energy time
    % ------------------------------------------------------------
    % save at maximum operating time
    fig = figure(2); fig.Position = [500 800 800 500];
    % save PDF
    plot(xpts,f_pred,"-.","LineWidth",5.0,"Color","#7E2F8E"); hold on;
    plot(xpts,f_kde,"-","LineWidth",5.0,"Color",[0 0 0 0.7]); hold off;
    title(sprintf("$t=%.2f$",curr_time), ...
        "FontSize",24,"FontName","Times New Roman", ...
        "Interpreter","latex",'fontweight','bold');
    % axes labels
    xlabel("Line 35-36 Energy","FontSize",30, ...
        "FontName","Times New Roman");
    ylabel("PDF","FontSize",30,"FontName","Times New Roman");
    ax = gca;
    ax.FontSize = 30; box on;
    ax = gca; ax.LineWidth = 2;
    % legend for lines
    legend(["Pred","KDE"],"FontSize",30,"Location","northeast");

    
    % save large plot
    if nn == max_energy_time_idx
        figure_name = "./fig/case57/line1/max_energy_time_PDF.png";
        % save figure
        exportgraphics(gcf,figure_name,"Resolution",300);
    end

    % plot Zoomed in plot
    fig = figure(3); fig.Position = [500 800 500 500];
    % find positions > failure level
    idx = find(xpts>failure_level,1)-50;
    plot(xpts(idx:end),f_pred(idx:end),"-.","LineWidth", ...
        6.0,"Color","#7E2F8E"); hold on;
    plot(xpts(idx:end),f_kde(idx:end),"-","LineWidth", ...
        6.0,"Color","black"); hold off;
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
    % shade area under KDE curve
    idx = find(xpts>failure_level,1);
    ar = area(xpts(idx:end),f_kde(idx:end),"FaceColor","red");
    ar.FaceAlpha = 0.4;
    hold off;

    % save small plot
    if nn == max_energy_time_idx
        figure_name = "./fig/case57/line1/max_energy_time_PDF_small.png";
        % save figure
        exportgraphics(gcf,figure_name,"Resolution",300);
    end


    % calculate probabilities and save to .mat
    if nn == max_energy_time_idx
        idx = find(xpts>failure_level,1)-5;
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
        writetable(T, "./fig/case57/line1/exceedance.txt");
    end

    % ------------------------------------------------------------
    % SAVE FIGURES
    % ------------------------------------------------------------
    % save at selected times 
%     save_figures = true;
%     if save_figures
%         if curr_time == 1.5 || curr_time == 3.0 || ... 
%                 curr_time == 4.5 || curr_time == 6.0
%             figure_name = strcat("./fig/CASE57_ROPDF_CDF_Time_", ...
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
%             title("Case 57","FontSize",18,"FontName","Times New Roman");
%             xlabel("Line 36-40 Energy","FontSize",18,"FontName","Times New Roman");
%             ylabel("CDF","FontSize",18,"FontName","Times New Roman");
%             xlim([0 140]);
%             ylim([0 1.0]);
%             ax = gca;
%             ax.FontSize = 18;
%             box on;
%             ax = gca;
%             ax.LineWidth = 2;
%            
%             % add more lines 
%             hold on;
%             if curr_time == 6.0
%                 legend(["t = 1.5", "", "t = 3.0", "", "t = 4.5", ...
%                     "", "t = 6.0", "Benchmark"], ...
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
%     end


    % ------------------------------------------------------------
end

%% Save computed solutions
fname = "./data/case57/line1_ropdf.mat";
save(fname,"f","xpts","mcro");

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
    
    figure(1);
    plot(xpts, f_kde(3:end-2, nn), "LineWidth", 2.0, "Color", "red");
end

%% Save MC solutions (for comparison, 11/08/2023)
fname = "./data/case57/line1_kde.mat";
save(fname,"f_kde","xpts","mcro");



%% Learn coffcients & solve 1d marginal PDE via lax-wendroff (line 2)
close all; 
rng('default');

% line is fixed and data simulated above
from_line = from_line2; 
to_line=to_line2; 

% adjust factor (Julia estimated: (36,40)=>rateA:20.27 p.u.
adjust_factor = 0.18;
mc_energy=mc_energy2*adjust_factor; 
mc_condexp_target=mc_target2*adjust_factor;
% set up pde domain
dx = 0.2;          % spatial step size
ng = 2;             % number of ghost cells
a0 = -5*dx;            % energy cannot be negative
b0 = max(mc_energy,[],"all")+1.0*std(mc_energy,[],"all"); % padded right boundary
nx = ceil((b0-a0)/dx);                      % number of nodes
xpts = linspace(a0+0.5*dx, b0-0.5*dx,nx)';  % column vector of cell centers
dx = xpts(2)-xpts(1);                       % updated dx if needed
xpts_e = [xpts-0.5*dx; xpts(end)+0.5*dx];   % cell edges for advection coeff

% allocate pde solution (IC from kernel density estimate)
f = zeros(nx+2*ng,nt);
f_ind = ng+1:nx+ng;     % indices of non-ghost cells

% compute failure level for plotting
start_idx = 40;
f0 = [squeeze(mc_energy(:,start_idx))];
bw = 0.9*min(std(f0), iqr(f0)/1.34)*(mc)^(-0.2);
f(f_ind,start_idx) = ksdensity(f0,xpts,'bandwidth',bw,"Support","positive","BoundaryCorrection","reflection");
figure(1);
plot(xpts, f(f_ind,start_idx),"LineWidth",1.5,"Color","black");

% find max mean operating energy
mean_energy = mean(mc_energy,1)';
max_energy = max(mean_energy);
max_energy_time_idx = find(mean_energy==max_energy);

% set failure level
failure_level = 20.27;
%% Begin RO-PDF

% reduced order MC trial numbers
mcro = 5000;
% time loop
for nn = start_idx+1:nt
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
    coeff = get_coeff(energy_data,response_data, ...
        xpts_e,"lin");

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
        % using same coeff., and then output solution on the coarser time
        % grid.
        
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

    % visualize learned coeffs, RO-PDF solution and exact solution from KDE
    fig=figure(1);
    fig.Position = [100 500 1600 400];
    
    % RO-PDF predictions
    set(gca,'linewidth',1.5, 'fontsize',20)
    f_pred = f(f_ind,nn);
    
    plot(xpts,f_pred,'-b','linewidth',2);
    xlabel('x')
    title('Marginal PDF');
    hold on;
    f0 = [squeeze(mc_energy(:,nn))];
    bw = 0.9*min(std(f0), iqr(f0)/1.34)*(mc)^(-0.2);
    f_kde = ksdensity(f0,xpts,'Support','positive', ...
        'BoundaryCorrection','reflection');
    plot(xpts,f_kde,"--", "Color", "black",'linewidth',2);
    hold off;
    
    % compute L^2 error
    tmp  =trapz(dx,(f_kde-f_pred).^2)/trapz(dx,f_kde.^2);
    disp(tmp)

    % ------------------------------------------------------------
    % Save at max mean energy time
    % ------------------------------------------------------------
    % save at maximum operating time
    %%
    fig = figure(2); fig.Position = [500 800 800 500];
    % save PDF
    plot(xpts,f_pred,"-.","LineWidth",5.0,"Color","#7E2F8E"); hold on;
    plot(xpts,f_kde,"-","LineWidth",5.0,"Color",[0 0 0 0.7]); hold off;
    title(sprintf("$t=%.2f$",curr_time), ...
        "FontSize",24,"FontName","Times New Roman", ...
        "Interpreter","latex",'fontweight','bold');
    % axes labels
    xlabel("Line 36-40 Energy","FontSize",30, ...
        "FontName","Times New Roman");
    ylabel("PDF","FontSize",30,"FontName","Times New Roman");
    ax = gca;
    ax.FontSize = 30; box on;
    ax = gca; ax.LineWidth = 2;
    % legend for lines
    legend(["Pred","KDE"],"FontSize",30,"Location","northeast");

    
    % save large plot
    if nn == max_energy_time_idx
        figure_name = "./fig/case57/line2/max_energy_time_PDF.png";
        % save figure
        exportgraphics(gcf,figure_name,"Resolution",300);
    end

    % plot Zoomed in plot
    fig = figure(3); fig.Position = [500 800 500 500];
    % find positions > failure level
    idx = find(xpts>failure_level,1)-5;
    plot(xpts(idx:end),f_pred(idx:end),"-.","LineWidth", ...
        6.0,"Color","#7E2F8E"); hold on;
    plot(xpts(idx:end),f_kde(idx:end),"-","LineWidth", ...
        6.0,"Color","black"); hold off;
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
    % shade area under KDE curve
    idx = find(xpts>failure_level,1);
    ar = area(xpts(idx:end),f_kde(idx:end),"FaceColor","red");
    ar.FaceAlpha = 0.4;
    hold off;
    % change units for y axis
    ax = gca; 
    ax.YAxis.Exponent = -2;


    % save small plot
    if nn == max_energy_time_idx
        figure_name = "./fig/case57/line2/max_energy_time_PDF_small.png";
        % save figure
        exportgraphics(gcf,figure_name,"Resolution",300);
    end

    %%
    % calculate probabilities and save to .mat
    if nn == max_energy_time_idx
        idx = find(xpts>failure_level,1)-5;
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
        writetable(T, "./fig/case57/line2/exceedance.txt");
    end

    


    % ------------------------------------------------------------
end



%% Save computed solutions
fname = "./data/case57/line2_ropdf.mat";
save(fname,"f","xpts","mcro");


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
    
    figure(1);
    plot(xpts, f_kde(3:end-2, nn), "LineWidth", 2.0, "Color", "red");
end

%% Save MC solutions (for comparison, 11/08/2023)
fname = "./data/case57/line2_kde.mat";
save(fname,"f_kde","xpts","mcro");









