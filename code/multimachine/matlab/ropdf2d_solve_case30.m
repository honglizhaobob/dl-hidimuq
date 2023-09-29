%     Hongli Zhao, Givens Associate, Argonne Nat. Lab.,
%     CAM PhD Student, UChicago
%     Last updated: Aug 16, 2023

% Solving 2d joint RO-PDF equation of energies of two lines
% Define simulation parameters
%%
clear; rng('default');
%% Load data
% Data should already have been simulated, see `ropdf_solve_case9.m`
fname = "./data/case30_mc_data.mat";
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
%%
% Compute energy for two lines connected to the same machine
from_line1 = 6; to_line1 = 7;
from_line2 = 6; to_line2 = 9;

%% Learn coffcients & solve 2d marginal PDE by corner transport
close all; 
rng('default');

% line is fixed and data simulated above
from_line1; to_line1; mc_energy1; mc_target1;
from_line2; to_line2; mc_energy2; mc_target2;

% set up pde domain
dx = 0.02; 
dy = dx*2;    % spatial step size
ng = 2;             % number of ghost cells

% left right boundaries
x_min = -10*dx; x_max = 4.5;
nx = ceil((x_max-x_min)/dx);
% top bottom boundaries
y_min = -10*dy; y_max = 4.0;
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

% allocate solution 
p = zeros(nx+4,ny+4,nt);
p_kde = zeros(nx+4,ny+4,nt);
start_idx = 1;
% I.C. from KDE (defined on cell center)
tmp=ksdensity([mc_energy1(:,start_idx) mc_energy2(:,start_idx)], [Xg(:) Yg(:)]);
tmp=reshape(tmp,[nx+4,ny+4]);
p(:,:,start_idx)=tmp;
% visualize
figure(1);
surf(Xg,Yg,p(:,:,start_idx)); view([90,90,90]);
title("IC"); xlabel("x"); ylabel("y"); zlabel("p(x,y,0)");

%% KDE solution
for nn = start_idx+1:nt
    tmp=ksdensity([mc_energy1(:,nn) mc_energy2(:,nn)], [Xg(:) Yg(:)]);
    tmp=reshape(tmp,[nx+4,ny+4]);
    p_kde = tmp(3:end-2,3:end-2);
    p_kde = p_kde/trapz(dy,trapz(dx,p_kde));
    figure(1);
    surf(p_kde);
    view([90,90,90]); 
    % save
    p2(3:end-2,3:end-2,nn)=p_kde;
end

%% Time loop
all_l2_err = [];
all_xmean_pred = [];
all_xmean_kde = [];
all_ymean_pred = [];
all_ymean_kde = [];
all_covariance_pred = [];
all_covariance_kde = [];
all_mutual_info = [];
% reduced order MC trials
mcro = 5000;
% mode for coefficient regression
mode = "lowess";
for nn = start_idx+1:nt
    disp(nn)
    % estimate coefficients on cell edges
    x_data = squeeze(mc_energy1(1:mcro,nn-1));
    y_data = squeeze(mc_energy2(1:mcro,nn-1));
    
    % estimate coefficients
    % d/dx coefficient
    response_x_data = squeeze(mc_target1(1:mcro,nn-1));
    coeff1 = get_coeff2d(x_data,y_data,response_x_data,xpts_e,ypts_e,mode);

    % d/dy coefficient
    response_y_data = squeeze(mc_target2(1:mcro,nn-1));
    coeff2 = get_coeff2d(x_data,y_data,response_y_data,xpts_e,ypts_e,mode);

    % modify time step size based on cfl
    tmp=(max(max(abs(coeff1/dx)))+max(max(abs(coeff2/dy))));
    if tmp==0
        dt2 = dt;
    else
        dt2=1/tmp;
    end

    % time stepping (if refined step size is smaller, use fine timestep
    % and output at coarse step), otherwise, use original time ste
    if dt2>=dt
        % step with dt
        p(:,:,nn) = transport(p(:,:,nn-1),coeff1,coeff2,dx,dy,dt);
    else
        % step with dt2 (rounded so dt2 divides dt), and step by dt2 until next dt
        nt_temp = ceil(dt/dt2)+1; 
        dt2 = dt/(nt_temp - 1);
        ptmp = p(:,:,nn-1);
        if dt2==0 || isnan(dt2)
            error('dt0 = 0 or NaN')
        end
        % take adaptive time steps

        % for interpolation, estimate coefficients of the next time step
        x_data_next = squeeze(mc_energy1(1:mcro,nn));
        y_data_next = squeeze(mc_energy2(1:mcro,nn));
        response_x_data_next = squeeze(mc_target1(1:mcro,nn));
        response_y_data_next = squeeze(mc_target2(1:mcro,nn));
        coeff1_next = get_coeff2d(x_data_next,y_data_next,response_x_data_next, ...
            xpts_e,ypts_e,mode);
        coeff2_next = get_coeff2d(x_data_next,y_data_next,response_y_data_next, ...
            xpts_e,ypts_e,mode);
        t0 = tt(nn-1); 
        t1 = tt(nn);
        for ll=2:nt_temp
            % linearly interpolate the coefficients in time
            t = t0+dt2*(ll-1);
            coeff1_lininterp = ((t1-t)/dt)*coeff1+((t-t0)/dt)*coeff1_next;
            coeff2_lininterp = ((t1-t)/dt)*coeff2+((t-t0)/dt)*coeff2_next;
            ptmp = transport(ptmp,coeff1_lininterp,coeff2_lininterp,dx,dy,dt2);
        end
        % store at coarse time step
        p(:,:,nn) = ptmp;
    end

    % process predictions
    tmp = p(:,:,nn);
    tmp(tmp<0.0) = abs(tmp(tmp<0.0));
    tmp = tmp / trapz(dy,trapz(dx,tmp));
    p(:,:,nn) = tmp;

    % validate PDE solution
    if max(max(abs(p(:,:,nn))))>1e5
        error('PDE blows up');
    end
    if any(any(isnan(p(:,:,nn))))
        error('PDE has NaN values');
    end

    % visualize
    fig=figure(1);
    fig.Position = [100 500 1600 400];
    subplot(1,2,1);
    % plot predicted (RO-PDF)
    p_pred = p(3:end-2,3:end-2,nn);

    surf(p_pred);
    view([90,90,90]); 
    subplot(1,2,2);
    % plot exact (bivariate kde)
    tmp=ksdensity([mc_energy1(:,nn) mc_energy2(:,nn)], [Xg(:) Yg(:)]);
    tmp=reshape(tmp,[nx+4,ny+4]);
    p_kde = tmp(3:end-2,3:end-2);
    p_kde = p_kde/trapz(dy,trapz(dx,p_kde));
    surf(p_kde);
    view([90,90,90]); 

    % record relative error in L^2(R^2)
    tmp = trapz(dx, trapz(dy, (p_pred-p_kde).^2));
    tmp2 = trapz(dx, trapz(dy, p_kde.^2));
    tmp = tmp/tmp2;
    disp(tmp)
    all_l2_err = [all_l2_err, tmp];

    % compute marginal means and covariance from joint PDF
    p_x_marg_pred = reshape(trapz(ypts(3:end-2),p_pred'),[],1);
    p_y_marg_pred = reshape(trapz(xpts(3:end-2)',p_pred),[],1);
    p_x_marg_kde = reshape(trapz(ypts(3:end-2),p_kde'),[],1);
    p_y_marg_kde = reshape(trapz(xpts(3:end-2)',p_kde),[],1);

    % marginal mean in x
    xmean_pred = trapz(dx,xpts(3:end-2).*p_x_marg_pred);
    all_xmean_pred = [all_xmean_pred xmean_pred];
    xmean_kde = trapz(dx,xpts(3:end-2).*p_x_marg_kde);
    all_xmean_kde = [all_xmean_kde xmean_kde];

    % marginal mean in y
    ymean_pred = trapz(dy,ypts(3:end-2).*p_y_marg_pred);
    all_ymean_pred = [all_ymean_pred ymean_pred];
    ymean_kde = trapz(dy,ypts(3:end-2).*p_y_marg_kde);
    all_ymean_kde = [all_ymean_kde ymean_kde];

    % covariance
    x_demean_pred = reshape(xpts(3:end-2),[],1)-xmean_pred;
    y_demean_pred = reshape(ypts(3:end-2),1,[])-ymean_pred;
    prod_pred = x_demean_pred * y_demean_pred;
    
    x_demean_kde = reshape(xpts(3:end-2),[],1)-xmean_kde;
    y_demean_kde = reshape(ypts(3:end-2),1,[])-ymean_kde;
    prod_kde = x_demean_kde * y_demean_kde;

    % integrate with joint pdf to get E[(X-EX)*(Y-EY)]
    cov_pred = trapz(dy,trapz(dx,prod_pred.*p_pred));
    all_covariance_pred = [all_covariance_pred cov_pred];

    % estimate covariance from KDE PDF
    %cov_kde = trapz(dy,trapz(dx,prod_kde.*p_kde));
    % estimate covariance directly from data
    cov_kde = cov(mc_energy1(:,nn),mc_energy2(:,nn));
    cov_kde = cov_kde(1,2);
    all_covariance_kde = [all_covariance_kde cov_kde];
end
%% Save computed solutions
save("./data/CASE30_2d_ROPDF_Sol.mat", "xpts","ypts","tt","p");
%% Plot estimated moments
figure(1);
plot(tt(1:length(all_xmean_pred)), all_xmean_pred, "LineWidth", 2.0, "Color", "red"); 
hold on; plot(tt(1:length(all_xmean_pred)),all_xmean_kde, "--", "LineWidth", ...
    2.0, "Color", "blue")
legend(["Pred", "True"]);
title("Estimated first moments in X");

figure(2);
plot(tt(1:length(all_xmean_pred)), all_xmean_kde, "LineWidth", 2.0, "Color", "red"); 
hold on; plot(tt(1:length(all_xmean_pred)),all_xmean_kde, "--", "LineWidth", ...
    2.0, "Color", "blue")
legend(["Pred", "True"]);
title("Estimated first moments in Y");

figure(3);
plot(tt(1:length(all_xmean_pred)), all_covariance_pred, "LineWidth", 3.0, "Color", "black"); 
hold on; plot(tt(1:length(all_xmean_pred)),all_covariance_kde, "--", "LineWidth", ...
    2.0, "Color", "blue")
legend(["Pred", "True"]);
title("Estimated covariance (X, Y)");

figure(4);
plot(tt(1:length(all_xmean_pred)), all_l2_err, "LineWidth", 3.0, "Color", "black"); 
title("Relative L^2 error");




