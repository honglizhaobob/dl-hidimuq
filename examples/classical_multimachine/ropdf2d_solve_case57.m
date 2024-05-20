%     Hongli Zhao, Givens Associate, Argonne Nat. Lab.,
%     CAM PhD Student, UChicago
%     Last updated: Aug 16, 2023

% Solving 2d joint RO-PDF equation of energies of two lines
% Define simulation parameters
%%
clear; rng('default');
%% Load data
% Data should already have been simulated, see `ropdf_solve_case9.m`
fname = "./data/case57_mc_data.mat";
if ~isfile(fname)
    error("run the 1d file to generate data. ")
else
    load(fname);
    % time grid definition should agree with the 1d
    tf = 10.0;
    dt = 0.01;           % learn PDE coefficients in increments of dt
    tt = 0:dt:tf;        % coarse uniform time grid
    nt = length(tt);     % number of time steps
end

% Compute energy for two lines connected to the same machine
from_line1 = 35; to_line1 = 36;
from_line2 = 36; to_line2 = 40;

% line is fixed and data simulated above
from_line1; to_line1; mc_energy1; mc_target1;
from_line2; to_line2; mc_energy2; mc_target2;

% adjustment (see 1d file)
adjust_factor1 = 0.17;
adjust_factor2 = 0.18;

mc_energy1 = adjust_factor1*mc_energy1;
mc_target1 = adjust_factor1*mc_target1;

mc_energy2 = adjust_factor2*mc_energy2;
mc_target2 = adjust_factor2*mc_target2;

% set up pde domain
dx = 0.1; 
dy = 0.2;    % spatial step size
ng = 2;             % number of ghost cells

% left right boundaries
x_min = -dx; x_max = max(mc_energy1,[],'all')+1.0*std(mc_energy1,[],"all");
nx = ceil((x_max-x_min)/dx);

% top bottom boundaries
y_min = -5*dy; y_max = max(mc_energy2,[],'all')+1.0*std(mc_energy2,[],"all");
ny = ceil((y_max-y_min)/dy);
% cell centers
xpts = linspace(x_min+0.5*dx,x_max-0.5*dx,nx)';
ypts = linspace(y_min+0.5*dy,y_max-0.5*dy,ny)';
% pad with ghost cells
xpts = [xpts(1)-2*dx;xpts(1)-dx;xpts;xpts(end)+dx;xpts(end)+2*dx];
ypts = [ypts(1)-2*dy;ypts(1)-dy;ypts;ypts(end)+dy;ypts(end)+2*dy];

% left cell edges
xpts_e = xpts-0.5*dx; ypts_e = ypts-0.5*dy;

% mesh for cell edges
[Yge,Xge]=meshgrid(ypts_e,xpts_e);

% mesh for cell centers
[Yg,Xg]=meshgrid(ypts,xpts);

%% Compute KDE solution (benchmark)
% save benchmark PDF
fname = "./data/case57/joint_benchmark.mat";
if ~isfile(fname)
    p_bench = zeros(nx+4,ny+4,nt);
    for nn = 1:nt
        disp(nn)
        tmp=ksdensity([mc_energy1(:,nn) mc_energy2(:,nn)], [Xg(:) Yg(:)]);
        tmp=reshape(tmp,[nx+4,ny+4]);
        tmp = tmp(3:end-2,3:end-2);
        tmp = tmp/trapz(dy,trapz(dx,tmp));
        p_bench(3:end-2,3:end-2,nn)=tmp;
        figure(1);
        surf(Xge(3:end-2,3:end-2),Yge(3:end-2,3:end-2),tmp); shading interp;
        view(2);
    end
    save(fname, "p_bench", "dx", "dy", "dt", "nx", "ny", "nt");
else
    load(fname);
end
% for nn = 800:nt
%     disp(nn)
%     tmp=ksdensity([mc_energy1(1:1000,nn) mc_energy2(1:1000,nn)], [Xg(:) Yg(:)]);
%     tmp=reshape(tmp,[nx+4,ny+4]);
%     tmp = tmp(3:end-2,3:end-2);
%     tmp = tmp/trapz(dy,trapz(dx,tmp));
%     figure(1);
%     surf(Xge(3:end-2,3:end-2),Yge(3:end-2,3:end-2),tmp); shading interp;
%     view(2);
% end 
%%
fname = "./data/case57/joint_observer.mat";
if ~isfile(fname)
    % benchmark for MC convergence study
    p_bench = load("./data/case57/joint_benchmark.mat").p_bench;
    % number of reduced MC trials
    mcro = 2.^(9:13);
    nmcro = length(mcro);
    % allocate solution
    p_obs = zeros(nmcro,nx+4,ny+4,nt);
    % for each reduced MC trial, compute KDE
    for ii = 1:nmcro
        mcidx = mcro(ii);
        fprintf(">> Compute NMC = %d\n\n", mcidx);
        for nn = 1:nt
            disp(nn)
            tmp=ksdensity([mc_energy1(1:mcidx,nn) mc_energy2(1:mcidx,nn)], [Xg(:) Yg(:)]);
            tmp=reshape(tmp,[nx+4,ny+4]);
            tmp = tmp(3:end-2,3:end-2);
            tmp = tmp/trapz(dy,trapz(dx,tmp));
            p_obs(ii,3:end-2,3:end-2,nn)=tmp;
            figure(1);
            surf(Xge(3:end-2,3:end-2),Yge(3:end-2,3:end-2),tmp); shading interp;
            view(2);
        end
    end
    % save
    save(fname, "p_obs");
else
    load(fname);
end
%% Preprocess 2d coefficients

% number of reduced MC trials
mcro = 2.^(9:13);
nmcro = length(mcro);
fname = "./data/case57/joint_coeffs.mat";
if ~isfile(fname)
    % benchmark for MC convergence study
    p_bench = load("./data/case57/joint_benchmark.mat").p_bench;
    % number of reduced MC trials
    mcro = 2.^(9:13);
    nmcro = length(mcro);
    % allocate solution
    coeff1 = zeros(nmcro,nx+4,ny+4,nt);
    coeff2 = zeros(nmcro,nx+4,ny+4,nt);
    % for each reduced MC trial, compute KDE
    mode = "lowess";
    for ii = 1:nmcro
        mcidx = mcro(ii);
        fprintf(">> Compute NMC = %d\n\n", mcidx);
        for nn = 1:nt
            disp(nn)
            x_data = squeeze(mc_energy1(1:mcidx,nn));
            y_data = squeeze(mc_energy2(1:mcidx,nn));
            response_x_data = squeeze(mc_target1(1:mcidx,nn));
            response_y_data = squeeze(mc_target2(1:mcidx,nn));
            tmp1 = get_coeff2d(x_data,y_data,response_x_data,xpts_e,ypts_e,mode);
            tmp2 = get_coeff2d(x_data,y_data,response_y_data,xpts_e,ypts_e,mode);
            % save
            coeff1(ii,:,:,nn)=tmp1;
            coeff2(ii,:,:,nn)=tmp2;
            figure(1);
            scatter3(x_data,y_data,response_x_data); hold on;
            surf(Xge,Yge,tmp1); 
            hold off;
            figure(2);
            scatter3(x_data,y_data,response_y_data); hold on;
            surf(Xge,Yge,tmp2);
            hold off;
        end
    end
    % save
    save(fname, "coeff1", "coeff2");
else
    load(fname);
end

%% Solve PDE (nudged)

% number of reduced MC trials
mcro = 2.^(9:12);
nmcro = length(mcro);
% allocate memory
p = zeros(nmcro, nx+4,ny+4,nt);
% learning rate for nudging
lr = 1e+3;

% starting index
start_idx = 50;

for kk = 4
    tmp=ksdensity([mc_energy1(:,start_idx) mc_energy2(:,start_idx)], [Xg(:) Yg(:)]);
    tmp=reshape(tmp,[nx+4,ny+4]);
    p(kk,:,:,start_idx)=tmp;
    for nn = start_idx+1:nt
        disp(nn)
        c1 = squeeze(coeff1(kk,:,:,nn-1));
        c2 = squeeze(coeff2(kk,:,:,nn-1));
        % modify time step size based on cfl
        tmp=(max(max(abs(c1/dx)))+max(max(abs(c2/dy))));
        if tmp==0
            dt2 = dt;
        else
            dt2=1/tmp;
        end
    
        % time stepping (if refined step size is smaller, use fine timestep
        % and output at coarse step), otherwise, use original time ste
        if dt2>=dt
            % step with dt
            p(kk,:,:,nn) = transport(squeeze(p(kk,:,:,nn-1)),c1,c2,dx,dy,dt);
        else
            % step with dt2 (rounded so dt2 divides dt), and step by dt2 until next dt
            nt_temp = ceil(0.5*dt/dt2)+1; 
            dt2 = 0.5*dt/(nt_temp - 1);
            ptmp = squeeze(p(kk,:,:,nn-1));
            if dt2==0 || isnan(dt2)
                error('dt0 = 0 or NaN')
            end
            % take adaptive time steps
            for ll=2:nt_temp
                ptmp = transport(ptmp,c1,c2,dx,dy,dt2);
            end
            src2 = squeeze(p_obs(kk,:,:,nn));
            src1 = squeeze(p_obs(kk,:,:,nn-1));
            ptmp = (ptmp + 0.5*dt*lr*(src2+src1-ptmp))/(1+0.5*dt*lr);
            % store at coarse time step
            p(kk,:,:,nn) = ptmp;
        end
    
        % process predictions
        tmp = abs(squeeze(p(kk,:,:,nn)));
        tmp = tmp / trapz(dy,trapz(dx,tmp));
        p(kk,:,:,nn) = tmp;
        p_pred = squeeze(p(kk,3:end-2,3:end-2,nn));
    
        % visualize
        fig=figure(1);
        fig.Position = [100 500 1600 400];
        subplot(1,2,1);
        
        % plot predicted (RO-PDF)
        contourf(p_pred);
    
        subplot(1,2,2);
        % plot exact (bivariate kde)
        p_kde = squeeze(p_bench(3:end-2,3:end-2,nn));
        contourf(p_kde); 
    
        % record relative error in L^1(R^2)
        tmp = trapz(dy, trapz(dx, abs(p_pred-p_kde)));
        disp(tmp)
    end
end

%% Plot and save at t = 2.5, 5.0, 7.5, 10.0
fig=figure(1);
fig.Position = [100 500 1200 1000];

subplot(2,2,1);
idx = 200+1;
tmp = squeeze(p(4,3:end-2,3:end-2,idx)); 
s = surf(Xge(3:end-2,3:end-2),Yge(3:end-2,3:end-2),tmp,"EdgeAlpha",0.4);
view([10,10,1000]);
colormap turbo; 
title(sprintf("$t = %.2f$", tt(idx)), ...
    "Interpreter", "latex", "FontSize", 36);
ax = gca; 
ax.FontSize = 36; 
xlabel(sprintf("Line %d-%d",from_line1,to_line1), ...
    "FontSize",24,"FontName","Times");
ylabel(sprintf("Line %d-%d",from_line2,to_line2), ...
    "FontSize",24,"FontName","Times");

subplot(2,2,2);
idx = 400+1;
tmp = squeeze(p(4,3:end-2,3:end-2,idx));
surf(Xge(3:end-2,3:end-2),Yge(3:end-2,3:end-2),tmp,"EdgeAlpha",0.4); 
view([10,10,1000]);
colormap turbo; 
title(sprintf("$t = %.2f$", tt(idx)), ...
    "Interpreter", "latex", "FontSize", 36);
ax = gca; 
ax.FontSize = 36; 
xlabel(sprintf("Line %d-%d",from_line1,to_line1), ...
    "FontSize",24,"FontName","Times");
ylabel(sprintf("Line %d-%d",from_line2,to_line2), ...
    "FontSize",24,"FontName","Times");

subplot(2,2,3);
idx = 600+1;
tmp = squeeze(p(4,3:end-2,3:end-2,idx));
surf(Xge(3:end-2,3:end-2),Yge(3:end-2,3:end-2),tmp,"EdgeAlpha",0.4);
view([10,10,1000]);
colormap turbo; 
title(sprintf("$t = %.2f$", tt(idx)), ...
    "Interpreter", "latex", "FontSize", 24);
ax = gca; 
ax.FontSize = 36; 
xlabel(sprintf("Line %d-%d",from_line1,to_line1), ...
    "FontSize",24,"FontName","Times");
ylabel(sprintf("Line %d-%d",from_line2,to_line2), ...
    "FontSize",24,"FontName","Times");

subplot(2,2,4);
idx = 800+1;
tmp = squeeze(p(4,3:end-2,3:end-2,idx));
surf(Xge(3:end-2,3:end-2),Yge(3:end-2,3:end-2),tmp,"EdgeAlpha",0.4); 
view([10,10,1000]);
colormap turbo; 
title(sprintf("$t = %.2f$", tt(idx)), ...
    "Interpreter", "latex", "FontSize", 36);
ax = gca; 
ax.FontSize = 36; 
xlabel(sprintf("Line %d-%d",from_line1,to_line1), ...
    "FontSize",24,"FontName","Times");
ylabel(sprintf("Line %d-%d",from_line2,to_line2), ...
    "FontSize",24,"FontName","Times");


sgtitle('Case 57',"FontSize",50,"FontName","Times","FontWeight","bold"); 

% add common colorbar
h = axes(fig,'visible','off'); 
c = colorbar(h,'Position',[0.93 0.168 0.022 0.65]);
clim([0 1.0]);
colormap(c,'turbo');
c.LineWidth = 1.5;
c.FontSize = 24;
% save figure
fname = "./fig/case57/joint/predicted2d.png";
exportgraphics(fig,fname,"Resolution",300);

%% Save computed solutions
save("./data/case57/joint_nudged.mat", "p", "lr");

%% Load computed solutions
load("./data/case57/joint_nudged.mat", "p", "lr");
%% Compute tail probabilities

% compute max mean levels

% line 1
mean_energy1 = reshape(mean(mc_energy1,1),[],1);
max_mean_energy1 = max(mean_energy1);
max_energy_time_idx1 = find(mean_energy1==max_mean_energy1);

% line 2
mean_energy2 = reshape(mean(mc_energy2,1),[],1);
max_mean_energy2 = max(mean_energy2);
max_energy_time_idx2 = find(mean_energy2==max_mean_energy2);

% failure levels
failure_level1 = 16.33; failure_level2 = 20.27;

% compute tail probability curves
tail_pred = zeros(nt,1);
tail_kde = zeros(nt,1);
joint_pred = zeros(nt,1);
joint_kde = zeros(nt,1);
for nn = 1:nt
    disp(nn)
    idx1 = find(xpts>failure_level1,1);
    idx2 = find(ypts>failure_level2,1);
    [p_tail_pred,F_x_pred,F_y_pred,F_xy_pred] ...
        = tail_prob2d(squeeze(p(end-1,3:end-2,3:end-2,nn)), dx, dy);
    [p_tail_kde,F_x_kde,F_y_kde,F_xy_kde] ...
        = tail_prob2d(squeeze(p_obs(end-1,3:end-2,3:end-2,nn)), dx, dy);
    tmp1 = 1-F_xy_pred(idx1,idx2);
    tmp2 = 1-F_xy_kde(idx1,idx2);
    %figure(1);
    %surf(Xge(3:end-2,3:end-2),Yge(3:end-2,3:end-2),1-F_xy_kde); shading interp;
    %view(2);
    joint_pred(nn) = tmp1;
    joint_kde(nn) = tmp2;
    % predicted
    tmp1 = (1-F_x_pred(idx1))+(1-F_y_pred(idx2))-(1-F_x_pred(idx1))*(1-F_y_pred(idx2));
    tail_pred(nn) = tmp1;
    % mc
    tmp2 = (1-F_x_kde(idx1))+(1-F_y_kde(idx2))-(1-F_x_kde(idx1))*(1-F_y_kde(idx2));
    tail_kde(nn) = tmp2;
end

% print max tail probabilities
format long
disp(max(tail_pred));
disp(max(tail_kde));

%%
% print tail probabilities estimated using product measure at the same
% reported time (ROPDF)
[~, idx] = max(tail_kde);
% load 1d ROPDFs
tmp1 = load("./data/case57/line1_ropdf.mat");
tmp2 = load("./data/case57/line2_ropdf.mat");
xpts = tmp1.xpts;
ypts = tmp2.xpts;
dx = xpts(2)-xpts(1);
dy = ypts(2)-ypts(1);
% form product measure
line1_marginal = tmp1.f(:,idx); 
line2_marginal = tmp2.f(:,idx);
line1_marginal_cdf = cumtrapz(dx,line1_marginal);
line2_marginal_cdf = cumtrapz(dy,line2_marginal);
prod_pred = line1_marginal_cdf.*line2_marginal_cdf';
idx1 = find(xpts>failure_level1,1);
idx2 = find(ypts>failure_level2,1);
disp(1-prod_pred(idx1,idx2));

%%
% print tail probabilities estimated using product measure at the same
% reported time (MC)
[~, idx] = max(tail_kde);
% load 1d ROPDFs
tmp1 = load("./data/case57/line1_kde.mat");
tmp2 = load("./data/case57/line2_kde.mat");
xpts = tmp1.xpts;
ypts = tmp2.xpts;
dx = xpts(2)-xpts(1);
dy = ypts(2)-ypts(1);
% form product measure
line1_marginal = tmp1.f_kde(:,idx); 
line2_marginal = tmp2.f_kde(:,idx);
line1_marginal_cdf = cumtrapz(dx,line1_marginal);
line2_marginal_cdf = cumtrapz(dy,line2_marginal);
prod_pred = line1_marginal_cdf.*line2_marginal_cdf';
idx1 = find(xpts>failure_level1,1);
idx2 = find(ypts>failure_level2,1);
disp(1-prod_pred(idx1,idx2));


