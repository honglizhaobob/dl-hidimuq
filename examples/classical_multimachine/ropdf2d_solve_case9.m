%     Hongli Zhao, Givens Associate, Argonne Nat. Lab.,
%     CAM PhD Student, UChicago
%     Last updated: Aug 16, 2023

% Solving 2d joint RO-PDF equation of energies of two lines
% Define simulation parameters
%%
clear; rng('default');
%% Load data
% Data should already have been simulated, see `ropdf_solve_case9.m`
fname = "./data/case9_mc1d_coeff_data.mat";
if ~isfile(fname)
    error("run the 1d file to generate data. ")
else
    load(fname);
    tt = tt(:);
    nt = length(tt);
    dt = tt(2)-tt(1);
end
%%
% Compute energy for two lines connected to the same machine
from_line1 = 4; to_line1 = 9;
from_line2 = 7; to_line2 = 8;

%% Learn coffcients & solve 2d marginal PDE by corner transport
close all; 
rng('default');

% line is fixed and data simulated above
from_line1; to_line1; mc_energy1; mc_condexp_target1;
from_line2; to_line2; mc_energy2; mc_condexp_target2;

% set up pde domain
dx = 0.02; 
dy = dx;    % spatial step size
ng = 2;             % number of ghost cells

% left right boundaries
x_min = 0; x_max = max(mc_energy1,[],'all');
nx = ceil((x_max-x_min)/dx);
% top bottom boundaries
y_min = 0; y_max = max(mc_energy2,[],'all');
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

% allocate solution and KDE comparison
p = zeros(nx+4,ny+4,nt);
pkde = p;

% I.C. from KDE (defined on cell center)
tmp=ksdensity([mc_energy1(:,1) mc_energy2(:,1)], [Xg(:) Yg(:)]);
tmp=reshape(tmp,[nx+4,ny+4]);
p(:,:,1)=tmp;
pkde(:,:,1)=tmp;
% visualize
figure(1);
surf(Xg,Yg,p(:,:,1)); view([90,90,90]);
title("IC"); xlabel("x"); ylabel("y"); zlabel("p(x,y,0)");


%% Compute KDE solution of different resolution (for nudging)
fname = "./data/case9/joint_observer.mat";
if ~isfile(fname)
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
            surf(tmp); shading interp;
            view(2);
        end
    end
    % save
    save(fname, "p_obs");
else
    load(fname)
end
%%
% number of reduced MC trials
mcro = 2.^(9:13);
nmcro = length(mcro);
% allocate memory
p = zeros(nmcro, nx+4,ny+4,nt);
% learning rate for nudging
lr = 1e+3;

% starting index
start_idx = 1;

for kk = 4
    tmp=ksdensity([mc_energy1(:,start_idx) mc_energy2(:,start_idx)], [Xg(:) Yg(:)]);
    tmp=reshape(tmp,[nx+4,ny+4]);
    p(kk,:,:,start_idx)=tmp;
    for nn = start_idx+1:nt
        disp(nn)
        mode = "lin";       
        % estimate coefficients on cell edges
        x_data = squeeze(mc_energy1(1:mcro(kk),nn-1));
        y_data = squeeze(mc_energy2(1:mcro(kk),nn-1));

        % d/dx coefficient
        response_x_data = squeeze(mc_condexp_target1(1:mcro(kk),nn-1));
        c1 = get_coeff2d(x_data,y_data,response_x_data,xpts_e,ypts_e,mode);

        % d/dy coefficient
        response_y_data = squeeze(mc_condexp_target2(1:mcro(kk),nn-1));
        c2 = get_coeff2d(x_data,y_data,response_y_data,xpts_e,ypts_e,mode);

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
        tmp=ksdensity([mc_energy1(:,nn) mc_energy2(:,nn)], [Xg(:) Yg(:)]);
        tmp=reshape(tmp,[nx+4,ny+4]);
        tmp = tmp / trapz(dy,trapz(dx,tmp));
        tmp = tmp(3:end-2,3:end-2);
        contourf(tmp); 
    
        % record relative error in L^1(R^2)
        tmp = trapz(dy, trapz(dx, abs(p_pred-tmp)));
        disp(tmp)
    end
end

%% Plot and save at t = 2.5, 5.0, 7.5, 10.0
fig=figure(1);
fig.Position = [100 500 1200 1000];

subplot(2,2,1);
idx = 250+1;
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
idx = 500+1;
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
idx = 750+1;
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
idx = 1000+1;
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

sgtitle('Case 9',"FontSize",50,"FontName","Times","FontWeight","bold"); 

% add common colorbar
h = axes(fig,'visible','off'); 
c = colorbar(h,'Position',[0.93 0.168 0.022 0.65]);
clim([0 50]);
colormap(c,'turbo');
c.LineWidth = 1.5;
c.FontSize = 24;
% save figure
fname = "./fig/case9/joint/predicted2d.png";
exportgraphics(fig,fname,"Resolution",300);

%% Time loop
% reduced order MC trials
% mcro = 5000;
% for nn = 2:nt
%     disp(nn)
%     curr_time = nn*dt;
%     % estimate coefficients on cell edges
%     x_data = squeeze(mc_energy1(1:mcro,nn-1));
%     y_data = squeeze(mc_energy2(1:mcro,nn-1));
% 
%     % d/dx coefficient
%     response_x_data = squeeze(mc_condexp_target1(1:mcro,nn-1));
%     
%     % mode for coefficient regression
%     mode = "lin";       
%     coeff1 = get_coeff2d(x_data,y_data,response_x_data,xpts_e,ypts_e,mode);
% 
%     % d/dy coefficient
%     response_y_data = squeeze(mc_condexp_target2(1:mcro,nn-1));
%     coeff2 = get_coeff2d(x_data,y_data,response_y_data,xpts_e,ypts_e,mode);
% 
%     % modify time step size based on cfl
%     tmp=(max(max(abs(coeff1/dx)))+max(max(abs(coeff2/dy))));
%     if tmp==0
%         dt2 = dt;
%     else
%         dt2=1/tmp;
%     end
% 
%     % time stepping (if refined step size is smaller, use fine timestep
%     % and output at coarse step), otherwise, use original time ste
%     if dt2>=dt
%         % step with dt
%         p(:,:,nn) = transport(p(:,:,nn-1),coeff1,coeff2,dx,dy,dt);
%     else
%         % step with dt2 (rounded so dt2 divides dt), and step by dt2 until next dt
%         nt_temp = ceil(dt/dt2)+1; 
%         dt2 = dt/(nt_temp - 1);
%         ptmp = p(:,:,nn-1);
%         if dt2==0 || isnan(dt2)
%             error('dt0 = 0 or NaN')
%         end
%         % take adaptive time steps
%         for ll=2:nt_temp
%             ptmp = transport(ptmp,coeff1,coeff2,dx,dy,dt2);
%         end
%         % store at coarse time step
%         p(:,:,nn) = ptmp;
%     end
% 
%     % process predictions
%     tmp = p(:,:,nn);
%     tmp = tmp / trapz(dy,trapz(dx,tmp));
%     p(:,:,nn) = tmp;
%     % save KDE estimate
%     tmp=ksdensity([mc_energy1(:,nn) mc_energy2(:,nn)], [Xg(:) Yg(:)]);
%     tmp=reshape(tmp,[nx+4,ny+4]);
%     tmp = tmp / trapz(dy,trapz(dx,tmp));
%     pkde(:,:,nn) = tmp;
%     % compute L^2 error
%     disp(trapz(dy,trapz(dx,(p(:,:,nn)-pkde(:,:,nn)).^2)));
% end
%% Save computed solutions and KDE comparison or load
fname = "./data/CASE9_2d_ROPDF_Sol.mat";
if ~isfile(fname)
    save(fname,"xpts","ypts","tt","p","pkde");
else
    load(fname);
    nt = length(tt);
    dx = xpts(2)-xpts(1);
    dy = ypts(2)-ypts(1);
    all_joint_pred = zeros(nt,1);
    all_joint_kde = zeros(nt,1);
    all_prod_pred = zeros(nt,1);
    all_prod_kde = zeros(nt,1);
    for i = 1:nt
        i
        % compute tail probabilities at threshold
        failure_level1 = 1.0; failure_level2 = 1.0;
        idx1 = find(xpts>failure_level1,1);
        idx2 = find(ypts>failure_level2,1);
        [p_tail_pred,F_x_pred,F_y_pred,F_xy_pred] ...
            = tail_prob2d(p(3:end-2,3:end-2,i), dx, dy);
        [p_tail_kde,F_x_kde,F_y_kde,F_xy_kde] ...
            = tail_prob2d(pkde(3:end-2,3:end-2,i), dx, dy);
        tmp1 = 1-F_xy_pred(idx1,idx2);
        tmp2 = 1-F_xy_kde(idx1,idx2);
        
        all_joint_pred(i) = tmp1;
        all_joint_kde(i) = tmp2;

        % compute either-or probabilities using product measure
        all_prod_pred(i) = (1-F_x_pred(idx1))+(1-F_y_pred(idx2))-(1-F_x_pred(idx1))*(1-F_y_pred(idx2));
        all_prod_kde(i) = (1-F_x_kde(idx1))+(1-F_y_kde(idx2))-(1-F_x_kde(idx1))*(1-F_y_kde(idx2));
    end
    format long
    disp(all_joint_pred(245)); 
    disp(all_joint_kde(245)); 
    disp(all_prod_pred(245)); 
    disp(all_prod_kde(245)); 
end



