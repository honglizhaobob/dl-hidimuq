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
dx = 0.01; 
dy = dx*2;    % spatial step size
ng = 2;             % number of ghost cells

% left right boundaries
x_min = 0; x_max = 1.5;
nx = ceil((x_max-x_min)/dx);
% top bottom boundaries
y_min = 0; y_max = 2;
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
% I.C. from KDE (defined on cell center)
tmp=ksdensity([mc_energy1(:,1) mc_energy2(:,1)], [Xg(:) Yg(:)]);
tmp=reshape(tmp,[nx+4,ny+4]);
p(:,:,1)=tmp;
% visualize
figure(1);
surf(Xg,Yg,p(:,:,1)); view([90,90,90]);
title("IC"); xlabel("x"); ylabel("y"); zlabel("p(x,y,0)");

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
mcro = 2000;
figure_time = 1.5;
for nn = 2:nt
    disp(nn)
    curr_time = nn*dt;
    % estimate coefficients on cell edges
    x_data = squeeze(mc_energy1(1:mcro,nn-1));
    y_data = squeeze(mc_energy2(1:mcro,nn-1));
    % d/dx coefficient
    response_x_data = squeeze(mc_condexp_target1(1:mcro,nn-1));
    
    % mode for coefficient regression
    mode = "lowess";       % linear regression
    %mode = "lowess";    % LOWESS smoothing
    coeff1 = get_coeff2d(x_data,y_data,response_x_data,xpts_e,ypts_e,mode);

    % d/dy coefficient
    response_y_data = squeeze(mc_condexp_target2(1:mcro,nn-1));
    coeff2 = get_coeff2d(x_data,y_data,response_y_data,xpts_e,ypts_e,mode);

    % zero out coefficients outside of main support
    %coeff1 = clip_coeff(coeff1,Xg,Yg,x_data,y_data);
    %coeff2 = clip_coeff(coeff2,Xg,Yg,x_data,y_data);

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
        for ll=2:nt_temp
            ptmp = transport(ptmp,coeff1,coeff2,dx,dy,dt2);
        end
        % store at coarse time step
        p(:,:,nn) = ptmp;
    end

    % process predictions
    tmp = p(:,:,nn);
    tmp(tmp<0.0) = abs(tmp(tmp<0.0));
    tmp = tmp / trapz(dy,trapz(dx,tmp));
    p(:,:,nn) = tmp;

    if max(max(abs(p(:,:,nn))))>1e5
        error('PDE blows up');
    end
    if any(any(isnan(p(:,:,nn))))
        error('PDE has NaN values');
    end

    p_pred = p(3:end-2,3:end-2,nn);
    tmp=ksdensity([mc_energy1(:,nn) mc_energy2(:,nn)], [Xg(:) Yg(:)]);
    tmp=reshape(tmp,[nx+4,ny+4]);
    p_kde = tmp(3:end-2,3:end-2);
    p_kde = p_kde/trapz(dy,trapz(dx,p_kde));
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
    %  cov_kde = trapz(dy,trapz(dx,prod_kde.*p_kde));
    % estimate covariance directly from data
    cov_kde = cov(mc_energy1(:,nn),mc_energy2(:,nn));
    cov_kde = cov_kde(1,2);
    all_covariance_kde = [all_covariance_kde cov_kde];

    save_figures = false;
    if save_figures
        % ----------------------------------------------------------------------
        % save figures
        % ----------------------------------------------------------------------
        %figure_time = 1.5;
        f1name = "./fig/CASE9_JointPDF.png";
        f2name = "./fig/CASE9_CondCDF.png";
        f3name = "./fig/CASE9_Condexp1.png";
        f4name = "./fig/CASE9_Condexp2.png";
    
        % ---------------------
        fig = figure(1);
        fig.Position = [100 500 1200 1000];
        % Surface plot of predicted 2d tail probability
        surf(Xg(3:end-2,3:end-2),Yg(3:end-2,3:end-2), ...
            tail_prob2d(p_pred,dx,dy),"EdgeAlpha",0.2); 
        xlabel("Line 4-9","FontSize",60); 
        ylabel("Line 7-8","FontSize",60);
        zlabel("Joint PDF","FontSize",60);
        view([60,50,60]);
        title("t = 1.50","FontSize",60);
        set(gca,"FontSize",50)
    
        if curr_time == figure_time
            exportgraphics(fig,f1name,"Resolution",300);
        end
        % ---------------------
        % Plot of conditional PDF and compare with KDE
        yidx = 21+2;     % 2 ghost cells 
        y1 = ypts(yidx); % fixed y value
        % integrate in x
        p_y_marg_pred; p_y_marg_kde;
        y1_marg = p_y_marg_pred(yidx);
        % predicted conditional density: y1
        x_cond_y1_pred = p_pred(yidx,:)./p_y_marg_pred(yidx);
        % KDE conditional density
        x_cond_y1_kde = p_kde(yidx,:)./p_y_marg_kde(yidx);
        
        % plot conditional PDF: f(x|y=y1) = f(x,y1)/f(11)
        fig = figure(2);
        fig.Position = [200 500 1600 1000];
    
        subplot(1,3,1);
        plot(xpts(3:end-2),x_cond_y1_pred,"Color","blue","LineWidth",2.5);
        hold on;
        plot(xpts(3:end-2),x_cond_y1_kde,"--","LineWidth",5.0,"Color", ...
            [0 0 0 0.5]);
        xlabel("$u_1$","Interpreter","latex","FontSize",30);
        ylabel("Conditional PDF", ...
            "Interpreter","latex","FontSize",30);
        title("$u_2=0.4$","Interpreter","latex","FontSize",30);
        legend(["ROPDF", "KDE"],"Location","northeast","FontSize",30);
        hold off;
        set(gca,"FontSize",30)
    
        % Plot of conditional PDF and compare with KDE
        yidx = 31+2;     % 2 ghost cells 
        y1 = ypts(yidx); % fixed y value
        % integrate in x
        p_y_marg_pred; p_y_marg_kde;
        y1_marg = p_y_marg_pred(yidx);
        % predicted conditional density: y1
        x_cond_y1_pred = p_pred(yidx,:)./p_y_marg_pred(yidx);
        % KDE conditional density
        x_cond_y1_kde = p_kde(yidx,:)./p_y_marg_kde(yidx);
        
        % plot conditional PDF: f(x|y=y1) = f(x,y1)/f(11)
        subplot(1,3,2);
        plot(xpts(3:end-2),x_cond_y1_pred,"Color","red","LineWidth",2.5);
        hold on;
        plot(xpts(3:end-2),x_cond_y1_kde,"--","LineWidth",5.0,"Color", ...
            [0 0 0 0.5]);
        xlabel("$u_1$","Interpreter","latex","FontSize",30);
        title("$u_2=0.6$","Interpreter","latex","FontSize",30);
        hold off;
        set(gca,"FontSize",30)
    
        % Plot of conditional PDF and compare with KDE
        yidx = 41+2;     % 2 ghost cells 
        y1 = ypts(yidx); % fixed y value
        % integrate in x
        p_y_marg_pred; p_y_marg_kde;
        y1_marg = p_y_marg_pred(yidx);
        % predicted conditional density: y1
        x_cond_y1_pred = p_pred(yidx,:)./p_y_marg_pred(yidx);
        % KDE conditional density
        x_cond_y1_kde = p_kde(yidx,:)./p_y_marg_kde(yidx);
        
        % plot conditional PDF: f(x|y=y1) = f(x,y1)/f(11)
        subplot(1,3,3);
        plot(xpts(3:end-2),x_cond_y1_pred,"Color","magenta","LineWidth",2.5);
        hold on;
        plot(xpts(3:end-2),x_cond_y1_kde,"--","LineWidth",5.0,"Color", ...
            [0 0 0 0.5]);
        xlabel("$u_1$","Interpreter","latex","FontSize",30);
        title("$u_2=0.8$","Interpreter","latex","FontSize",30);
        hold off;
        set(gca,"FontSize",30);
        hold off;
    
        if curr_time == figure_time
            exportgraphics(fig,f2name,"Resolution",300);
        end
    
        % Plot coefficient 1
        fig = figure(3);
        fig.Position = [300 500 1200 1000];
        scatter3(x_data,y_data,response_x_data,10,"+","MarkerEdgeColor","black"); 
        hold on; surf(Xg,Yg,coeff1,"EdgeAlpha",0.2);
        xlabel("Line 4-9","FontSize",60); 
        ylabel("Line 7-8","FontSize",60);
        title("Advection in $U^{(1)}$", "Interpreter", "latex", "FontSize", 60, ...
            "FontWeight", "bold");
        hold off;
    
        set(gca,"FontSize",60)
    
        if curr_time == figure_time
            exportgraphics(fig,f3name,"Resolution",300);
        end
    
    
        % Plot coefficient 2
        fig = figure(4);
        fig.Position = [400 500 1200 1000];
        scatter3(x_data,y_data,response_y_data,10,"+","MarkerEdgeColor","black"); 
        hold on; surf(Xg,Yg,coeff2,"EdgeAlpha",0.2); 
        xlabel("Line 4-9","FontSize",60); 
        ylabel("Line 7-8","FontSize",60);
        title("Advection in $U^{(2)}$", "Interpreter", "latex", "FontSize", 60, ...
            "FontWeight", "bold");
        hold off;
    
        set(gca,"FontSize",60)
        if curr_time == figure_time
            exportgraphics(fig,f4name,"Resolution",300);
        end
    end
end
%% Save computed solutions
save("./data/CASE9_2d_ROPDF_Sol.mat", "xpts","ypts","tt","p");
%%

%% Measure different MC trials error
all_mc = [250, 1000, 2500, 5000, 7500, 10000];
n_trials = length(all_mc);
% store relative L^2 errors from KDE
all_errors = zeros(n_trials,nt-1);
for i = 1:n_trials
    % reduced order MC trial numbers
    mcro = all_mc(i);
    disp(strcat("===> MC = ", num2str(mcro)));

    for nn = 2:nt
        disp(nn)
        curr_time = nn*dt;
        % estimate coefficients on cell edges
        x_data = squeeze(mc_energy1(1:mcro,nn-1));
        y_data = squeeze(mc_energy2(1:mcro,nn-1));
        % d/dx coefficient
        response_x_data = squeeze(mc_condexp_target1(1:mcro,nn-1));
        
        % mode for coefficient regression
        %mode = "lin";       % linear regression
        mode = "lowess";    % LOWESS smoothing
        coeff1 = get_coeff2d(x_data,y_data,response_x_data,xpts_e,ypts_e,mode);
    
        % d/dy coefficient
        response_y_data = squeeze(mc_condexp_target2(1:mcro,nn-1));
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
            for ll=2:nt_temp
                ptmp = transport(ptmp,coeff1,coeff2,dx,dy,dt2);
            end
            % store at coarse time step
            p(:,:,nn) = ptmp;
        end
    
        % process predictions
        tmp = p(:,:,nn);
        tmp(tmp<0.0) = abs(tmp(tmp<0.0));
        tmp = tmp / trapz(dy,trapz(dx,tmp));
        p(:,:,nn) = tmp;
    
        if max(max(abs(p(:,:,nn))))>1e5
            error('PDE blows up');
        end
        if any(any(isnan(p(:,:,nn))))
            error('PDE has NaN values');
        end
    
        p_pred = p(3:end-2,3:end-2,nn);
        tmp=ksdensity([mc_energy1(:,nn) mc_energy2(:,nn)], [Xg(:) Yg(:)]);
        tmp=reshape(tmp,[nx+4,ny+4]);
        p_kde = tmp(3:end-2,3:end-2);
        p_kde = p_kde/trapz(dy,trapz(dx,p_kde));
        % record relative error in L^2(R^2)
        tmp = trapz(dx, trapz(dy, (p_pred-p_kde).^2));
        tmp2 = trapz(dx, trapz(dy, p_kde.^2));
        tmp = tmp/tmp2;
        disp(tmp)
        all_errors(i,nn)=tmp;
    end
end
%%
% Linear regression
%save("./data/CASE9_2d_Lin_ConvStudy.mat", "all_errors", "all_mc");
% Gaussian LLR
%save("./data/CASE9_2d_GLLR_ConvStudy.mat", "all_errors", "all_mc");

%% Plot and compare errors
lin_conv = load("./data/CASE9_2d_Lin_ConvStudy.mat");
llr_conv = load("./data/CASE9_2d_GLLR_ConvStudy.mat");
figure(1);
all_mc = [250,1000,2500,5000,7500,10000];
plot(all_mc,trapz(dt,lin_conv.all_errors'),"LineWidth",2.5,"Color","blue");
hold on; 
plot(all_mc,trapz(dt,llr_conv.all_errors'),"--","LineWidth",2.5,"Color","black");
xlabel("Sample size","FontSize",25,"Interpreter","latex"); 
ylabel("$L^2$ error","FontSize",25,"Interpreter","latex");
title("Case 57: 2d","FontSize",30);
legend(["Linear","GLLR"],"FontSize",25);
set(gca,"FontSize",25)
exportgraphics(gca,"./fig/CASE9_ROPDF2d_Error.png","Resolution",200);


