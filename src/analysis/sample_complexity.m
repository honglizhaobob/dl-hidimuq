% Investigating sample complexity for all cases 9, 30, 57
% No line tripping, count complexity to reach average L^1 error
% 5% within benchmark (MC=2^15=32768), for t=0 to t=5.0, dt=5e-3.
% -> no line tripping, all PDFs are assumed to be in some sto-
% chastic equilibrium; PDF for all lines should be close to 
% each other (perhaps standardize first) by some energy distance/
% discrepancy measure. We do so to make an argument that the joint
% data distribution is of low complexity, such that the conditional
% expectations are simple/requires little samples to estimate. We 
% leave theoretical convergence study to future work.

% Added: 11/08/2023
clear; clc; rng("default");
%% Burn-in samples for each case 
case_nums = [9, 30, 57];
num_cases = length(case_nums);

% number of MC trials
nmc = 2^15;
% Make sure Matpower files added to path 
run("./matpower7.1/startup.m");
for c = 1:num_cases
    fprintf("> Beginning case %d\n", case_nums(c));
    case_name = sprintf("case%d.m", case_nums(c));
    % choose case from matpower7.1/data
    [dist, Pm, amat, vi, d0, success] = runpf_edited(case_name);
    % real and imaginary parts of admittance
    g = real(amat);
    % imag part
    b = imag(amat);

    n = length(d0);       % Number of buses
    H = ones(n,1);        % Inertia coefficients
    D = ones(n,1);        % Damping coefficients
    wr = 1;               % base speed

    % time grid for burn in
    tf = 50.0;               % final time for simulation
    dt = 5e-3;             % learn PDE coefficients in increments of dt
    time = 0:dt:tf;        % coarse uniform time grid
    tt = time;
    nt = length(time);     % number of time steps

    % Allocate random initial conditions:
    u0 = zeros(nmc,4*n);
    
    % Random voltages . (folded gaussian, mean at optimal vi given by matpower)
    sd_v = mean(vi)*0.01;
    v = abs(sd_v*randn(nmc,n) + reshape(vi,1,[]));
    
    % initialize speeds at equilibrium =wr
    u0_w = wr*ones(nmc,n);
    
    % initial angles at d0
    u0_d = repmat(d0',nmc,1);
    
    
    % Random initial conditions for OU noise
    theta = 1.0;                % drift parameter
    alpha = 0.05;               % diffusion parameter

    % define constant covariance matrix
    mode = "const";
    reactance_mat = [];
    susceptance_mat = [];
    R = cov_model(case_nums(c), mode, reactance_mat, susceptance_mat);
    C = chol(R)';
    eta0 = mvnrnd(zeros(n,1),(alpha^2)*R,nmc);
    
    % store in initial condition, ordering [v; w; delta; eta]
    u0(:, 1:n) = v;      
    u0(:, n+1:2*n) = u0_w;
    u0(:, 2*n+1:3*n) = u0_d;
    u0(:, 3*n+1:end) = eta0;

    % evaluate mean angles (over all MC, over all machines) to check convergence
    %mean_omega = zeros(n,nt);

    % burn initial conditions and save
    fname = sprintf("./data/case%d/complexity/burned_ic.mat",case_nums(c));
    burn_in = false;
    if ~isfile(fname) && burn_in
        % start burning
        for i = 1:nmc
            fprintf("> > > MC trial = %d\n", i);
            for j = 1:nt
                % step SDE
                u0(i,:)=classical_mc_step(dt,u0(i,:), ...
                    alpha,theta,C,H,D,Pm,wr,g,b);
                % compute norm of this MC trial
                tmp = u0(i,n+1:2*n)';
                % add to mean norm at this time
                %mean_omega(:,j) = mean_omega(:,j) + (1/nmc)*tmp;
            end
        end
        
        % after burning, save burned IC
        %save(fname, "u0", "mean_omega", "-v7.3");
        save(fname, "u0", "-v7.3");
        break;
    elseif isfile(fname)
        fprintf("> > Case %d IC already burned and stored. \n", ...
            case_nums(c));
        fprintf(">> Loading ...\n\n");
        load(fname);
    else
        % directly use random initial conditions.
    end
    
    fname = sprintf("./data/case%d/complexity/energy_samples.mat", ...
        case_nums(c));
    if ~isfile(fname)
        % given initial conditions, simulate trajectories for additional T
        % time, and record quantities of interest
        u0;
        % total number of lines
        all_lines = [];
        for i = 1:n
            for j = i+1:n
                if b(i,j) ~= 0
                    % add from and to line 
                    all_lines = [all_lines; i, j];
                end
            end
        end
    
        % for all lines, step SDE, compute QoI, and store
        num_lines = size(all_lines,1);
        energy = cell(num_lines,1);
        target = cell(num_lines,1);
    
        % time grid for simulation
        tf = 1.0;               
        dt = 5e-3;             
        time = 0:dt:tf;        
        tt = time;
        nt = length(time);     
    
        for l = 1:num_lines
            fprintf("> > Computing energy for line:%d=>(%d, %d)", ...
                l,all_lines(l,1),all_lines(l,2));
            % allocate array
            mc_energy = zeros(nmc,nt);
            mc_target = zeros(nmc,nt);
            for i = 1:nt
                i
                for j = 1:nmc
                    % step SDE
                    u0(j,:)=classical_mc_step(dt,u0(j,:),alpha,theta,C,H,D,Pm,wr,g,b);
                    from_line = all_lines(l,1);
                    to_line = all_lines(l,2);
                    % compute energy
                    mc_energy(j,i)=line_energy(b,from_line,to_line,u0(j,:));
                    % compute regression response
                    mc_target(j,i)=condexp_target(b,from_line,to_line,u0(j,:));
                end
            end
            % after filling in values, store in cell array
            energy{l} = mc_energy;
            target{l} = mc_target;
        end
        fprintf("> Finished generating data for case %d \n\n", case_nums(c));
        % after computing, save
        save(fname,"energy","target","tt","all_lines", "-v7.3");
    else
        fprintf("> Energy data already computed for case %d", ...
            case_nums(c));
    end
end

%% Compute distribution distances among all lines
clear; clc; rng("default");
% load computed energy
fname = "./data/case9/complexity/energy_samples.mat";
%fname = "./data/case30/complexity/energy_samples.mat";
%fname = "./data/case57/complexity/energy_samples.mat";
load(fname);
%%
% compute MC error convergence
mcro = 2.^(10:14);
num_lines = size(all_lines,1);
nt = length(tt);
dt = tt(2)-tt(1);
all_error_convergence = zeros(num_lines,length(mcro),nt);
for k = 1:num_lines
    data = energy{k};

    % determine grid for kernel density
    xmin = min(data,[],"all")-0.5*std(data,[],"all");
    xmax = max(data,[],"all")+0.5*std(data,[],"all");
    % use fixed size grid
    nx = 1000;
    xgrid = linspace(xmin,xmax,nx);
    dx = xgrid(2)-xgrid(1);
    for i = 1:length(mcro)
        i
        nmc = mcro(i);
        % compute L^1 error from benchmark
        for j = 1:nt
            j
            f0 = squeeze(data(:,j));
            fbench = ksdensity(f0,xgrid);
            figure(1);
            plot(xgrid,fbench);
            % reduce number of MC trials
            f0 = f0(1:nmc);
            fmc = ksdensity(f0,xgrid);
            % compute L^1 error and store
            tmp=trapz(dx,abs(fbench-fmc));
            all_error_convergence(k,i,j)=tmp;
            disp(tmp);
        end
    end
end

% integrate over all time
all_error_convergence_agg = mean(all_error_convergence,3);
%%
%fname = "./data/case9/complexity/mc_convergence.mat";
%fname = "./data/case30/complexity/mc_convergence.mat";
fname = "./data/case57/complexity/mc_convergence_lownoise.mat";
if ~isfile(fname)
    save(fname,"all_error_convergence");
end

%% Test RO-PDF convergence
mcro = [510, 1020, 2040, 4090, 8190];
num_lines = size(all_lines,1);
nt = length(tt);
dt = tt(2)-tt(1);
all_error_convergence = zeros(num_lines,length(mcro),nt);

% all grids for computing RO-PDF solution
all_grids = cell(num_lines,1);
all_sol = cell(num_lines,1);
for k = 1:num_lines
    data = energy{k};
    % determine grid for kernel density
    xmin = min(data,[],"all")-0.05*std(data,[],"all");
    xmax = max(data,[],"all")+0.05*std(data,[],"all");
    % use fixed size grid
    nx = 1000;
    xgrid = linspace(xmin,xmax,nx);
    dx = xgrid(2)-xgrid(1);
    disp(dx)
    all_grids{k} = xgrid;
    for i = 1:length(mcro)
        i
        nmc = mcro(i);
        
        % set up PDE
        ng = 2;
        % cell centers
        xpts = (xgrid+0.5*dx)';
        nx = length(xpts);
        % indexing for non-ghost cells
        f_ind = ng+1:nx+ng;

        xpts_e = [xpts-0.5*dx; xpts(end)+0.5*dx];
        % allocate PDF solution
        f = zeros(nx+2*ng,nt);
        f0 = [squeeze(data(:,1))];
        bw = 0.9*min(std(f0), iqr(f0)/1.34)*(nmc)^(-0.2);
        f(3:end-2,1) = ksdensity(f0,xpts,'bandwidth',bw);
        for nn = 2:nt
            nn
            % get X data (previous time)
            energy_data = squeeze(data(1:nmc,nn-1));
            
            % get Y data (previous time)
            response_data = squeeze(target{k}(1:nmc,nn-1));

            % transform energy data
            coeff = get_coeff(energy_data,response_data,xpts_e,"llr");
            figure(1);
            scatter(energy_data, response_data); 
            hold on;
            plot(xpts_e,coeff);
            hold off;
            %plot(xpts_e,coeff); hold off;
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
                f(3:end-2,nn) = f_temp(3:end-2);
                
            end    

            % compare with benchmark
            f0 = [squeeze(data(:,nn))];
            fbench = ksdensity(f0,xpts,"bandwidth",bw);
            % predicted solution
            fpred = f(f_ind,nn);
            tmp=trapz(dx,abs(fbench-fpred));
            
            all_error_convergence(k,i,nn) = tmp;
            disp(tmp);
        end
    end
end

%%
% integrate over all time
all_error_convergence_agg = mean(all_error_convergence,3);
%%
fname = "./data/case9/complexity/ropdf_convergence_lownoise.mat";
%fname = "./data/case30/complexity/ropdf_convergence.mat";
%fname = "./data/case57/complexity/ropdf_convergence.mat";
save(fname,"all_error_convergence");
%% Compare two curves
all_case_numbers = [9,30,57];
all_num_lines = [9,41,78];
all_samples_ropdf = zeros(3,1);
all_samples_mc = zeros(3,1);
for i = 1:length(all_case_numbers)
    case_number = all_case_numbers(i);
    % load case convergence in L^1
    mcfname = sprintf("./data/case%d/complexity/mc_convergence.mat",case_number);
    ropdffname = sprintf("./data/case%d/complexity/ropdf_convergence.mat",case_number);
    mcconv = load(mcfname).all_error_convergence;
    ropdfconv = load(ropdffname).all_error_convergence;
    mcro = 2.^(10:14);
    % integrate across time
    mcconv_agg = mean(mcconv(:,:,2:end),3);
    ropdfconv_agg = mean(ropdfconv(:,:,2:end),3);
    
    num_lines = size(mcconv_agg,1);
    disp(num_lines)
    % visualization
    visualize = true;
    if visualize
        % sum over all lines
        mcconv = mean(mcconv_agg,1);
        ropdfconv = mean(ropdfconv_agg,1);
        % visualize
        fig = figure(1);
        fig.Position = [100 100 1000 800];
        plot(log2(mcro), mcconv, "--*", "LineWidth", 3.5, "MarkerSize", 10); 
        hold on;
        plot(log2(mcro), ropdfconv, "-o", "LineWidth", 3.5, "MarkerSize", 10); 
        legend(["KDE (Case 9)","Pred (Case 9)", ...
            "KDE (Case 30)","Pred (Case 30)", ...
            "KDE (Case 57)","Pred (Case 57)"],"FontSize",30);
        tmp = 10:14;
        newxicklabs = cellstr(num2str(tmp(:), '2^{%d}'));
        set(gca,'XTick',tmp(:),'XTickLabel',newxicklabs,'TickLabelInterpreter','tex');
        ax = gca;
        ax.FontSize = 30;
        xlabel("Sample Size","FontSize",30,"FontName", "Times");
        ylabel("Average Error","FontSize",30,"FontName", "Times");
        hold on;

        if case_number == 57
            % save plot
            grid on;
            box on;
            set(gca,'linewidth',1);
            exportgraphics(gca,"./fig/error_convergence.png", ...
                "Resolution",300);
        end
    end
    % count samples by finding first indices in grid and sum up
    
    % 5% error from benchmark
    threshold = 0.01;

    % interpolate sample complexity on log scale
    mcro;
    lg2mcro = log2(mcro);
    count_samples_mc = 0;
    count_samples_ropdf = 0;
    for l = 1:num_lines
        % find first index such that for this line, error < threshold
        % if DNE, use largest sample
        idx = find(mcconv_agg(l,:)<threshold,1);
        idx2 = find(ropdfconv_agg(l,:)<threshold,1);
        % count KDE
        if isempty(idx)
            tmp = lg2mcro(end);
            % extrapolate
            slope = abs(mcconv_agg(l,end)-mcconv_agg(l,end-1));
            tmp = tmp+abs(threshold-mcconv_agg(l,end))/slope;
        elseif idx == 1
            tmp = lg2mcro(1);
        else
            % interpolate 
            slope = abs(mcconv_agg(l,idx)-mcconv_agg(l,idx-1));
            tmp = lg2mcro(idx-1)+abs(threshold-mcconv_agg(l,idx-1))/slope;
        end
        % count ROPDF
        if isempty(idx2)
            tmp2 = lg2mcro(end);
        elseif idx2 == 1
            tmp2 = lg2mcro(1);
        else
            % interpolate 
            slope = abs(ropdfconv_agg(l,idx2)-ropdfconv_agg(l,idx2-1));
            tmp2 = lg2mcro(idx2-1)+abs(threshold-ropdfconv_agg(l,idx2-1))/slope;
        end
        % increment samples
        count_samples_mc = count_samples_mc + 2^tmp;
        count_samples_ropdf = count_samples_ropdf + 2^tmp2;
    end
    all_samples_mc(i) = count_samples_mc;
    all_samples_ropdf(i) = count_samples_ropdf;
end

% Make 1% L^1 threshold sample size plot
fig = figure(4);
fig.Position = [100 100 1000 800];
plot(all_num_lines, log2(all_samples_ropdf), "-*", "Color", "green", ...
    "LineWidth", 3.5, "MarkerSize", 10.5); 
hold on;
plot(all_num_lines, log2(all_samples_mc), "--o", "Color", "blue", ...
    "LineWidth", 3.5, "MarkerSize", 10.5); 
ax = gca;
ax.XAxis.TickValues = all_num_lines;
tmp = 17:21;
tmp = tmp(:);
newyicklabs = cellstr(num2str(tmp(:), '2^{%d}'));
set(gca,'YTick',tmp(:),'YTickLabel',newyicklabs,'TickLabelInterpreter','tex');
ax.FontSize = 30;
xlabel("Number of Lines", "FontSize", 30, "FontName", "Times");
ylabel("Sample Size", "FontSize", 30, "FontName", "Times");
legend(["Pred", "KDE"],"Location","northwest","FontSize",30);
grid on;
box on;
set(gca,'linewidth',1);
exportgraphics(gca,"./fig/scalability.png","Resolution",300);
%% Compute exponents
tmpy = log2(all_samples_ropdf);
tmpx = log2(all_num_lines);
a1 = (tmpy(end)-tmpy(1))/(tmpx(end)-tmpx(1));

tmpy = log2(all_samples_mc);
a2 = (tmpy(end)-tmpy(1))/(tmpx(end)-tmpx(1));

