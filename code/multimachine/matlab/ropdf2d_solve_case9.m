%     Hongli Zhao, Givens Associate, Argonne Nat. Lab.,
%     CAM PhD Student, UChicago
%     Last updated: Aug 16, 2023

% Solving 2d joint RO-PDF equation of energies of two lines
% Define simulation parameters
%%
clear; rng('default');

% Make sure Matpower files added to path 
run("./matpower7.1/startup.m");

% choose case from matpower7.1/data
[Pm, g, b, vi, d0, success] = runpf_edited('case9.m');

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

% Governor constants 
R = 0.02;              % Droop
T1 = 0;                % Transient gain time
T2 = 0.1;              % Governor time constant

mc = 10000;             % Number of MC paths
tf = 50.0;               % final time for simulation
dt = 0.01;             % learn PDE coefficients in increments of dt
time = 0:dt:tf;        % coarse uniform time grid
tt = time;
nt = length(time);     % number of time steps

sig = 0.1*ones(n,1);   % SDE noise amplitudes for speeds
dt0 = 1e-2;            % time step for SDE --> must divide dt

if mod(dt,dt0)~=0
    error('dt0 does not divide dt')
end

%% Simulate Monte Carlo trajectories
if isfile("./data/case9_mc.mat")
    load("./data/case9_mc.mat");
else
    tic
    % Simulate Monte Carlo trajectories
    % Allocate random initial conditions:
    u0 = zeros(mc,4*n);
    % Random Initial speeds (Gaussian)
    mu_w = 1; sd_w = 0.1;
    u0_w = sd_w*randn(mc,n) + mu_w;
    
    % Random Initial angles Gaussian
    sd_d = 5.0 * pi/180.0; % sd_d = 10.0 degrees, mean around optimal d0
    u0_d = sd_d*randn(mc,n) + reshape(d0,1,[]);
    
    % Random voltages . (folded gaussian, mean at optimal vi)
    sd_v = mean(vi)*0.01;
    v = abs(sd_v*randn(mc,n) + reshape(vi,1,[]));
    
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
    %(3*N x nt x mc)
    paths_mc = classical_mc(mc,dt,nt,u0,alpha,theta,C,H,D,Pm,wr,g,b);
    toc
end
%%
% Compute energy for two lines connected to the same machine
tt = time';
from_line1 = 8; to_line1 = 9;
from_line2 = 8; to_line2 = 7;

assert(b(from_line1,to_line1)~=0.0 | g(from_line1,to_line1)~=0.0);
assert(b(from_line2,to_line2)~=0.0 | g(from_line2,to_line2)~=0.0);

% compute energy for each Monte Carlo trial at each time point
mc_energy1 = zeros(mc,nt);
mc_energy2 = zeros(mc,nt);
% compute expectation output for each Monte Carlo trial at each time point
mc_condexp_target1 = zeros(mc,nt);
mc_condexp_target2 = zeros(mc,nt);

for i =1:mc
    i
    for j = 1:nt
        % get solution
        u_i = reshape(paths_mc(i,:,j), [], 1);
        mc_energy1(i,j)=line_energy(b,from_line1,to_line1,u_i);
        mc_energy2(i,j) = line_energy(b,from_line2,to_line2,u_i);
        mc_condexp_target1(i,j)=condexp_target(b,from_line1,to_line1,u_i,wr);
        mc_condexp_target2(i,j)=condexp_target(b,from_line2,to_line2,u_i,wr);
    end
end
save("./data/case9_mc.mat", '-v7.3');

%% Visualize bivariate density of line energies
x_min = 0; x_max = 3;
y_min = 0; y_max = 6;
dx = 0.1; dy = 0.1;
xgrid = x_min:dx:x_max;
ygrid = y_min:dy:y_max;
[xmesh,ymesh] = meshgrid(ygrid,xgrid);
nx = length(xgrid); ny = length(ygrid);
all_pts2d = [];
for i=1:ny
    i
    for j=1:nx
        all_pts2d = [all_pts2d; [xgrid(j),ygrid(i)]];
    end
end
%%
stop_idx=1000;
all_correlations = zeros(stop_idx,1);
for i = 1:nt
    i
    if i>stop_idx
        break;
    end
    figure(1);
    h = mvksdensity([mc_energy1(:,i) mc_energy2(:,i)], ...
        all_pts2d);
    h = reshape(h,[nx,ny]);
    % compute correlation
    corr_mat = corr([mc_energy1(:,i) mc_energy2(:,i)]);
    % store correlation
    all_correlations(i)=corr_mat(1,2);
    % compute mass and normalize
    mass = trapz(dx,trapz(dy,h));
    disp(mass);
    h = h ./ mass;
    surf(xmesh,ymesh,h);
    view([90,90,90]); 
    xlabel("Line 1");
    ylabel("Line 2");
    zlabel("Density");
end
%%
figure(2);
plot(time(1:stop_idx),all_correlations,"LineWidth",2.5,"Color","black");
title("Correlation between Line 1 and Line 2 Energy (KDE)");
xlabel("Time"); ylabel("Correlation");
%% Learn coffcients & solve 2d marginal PDE by corner transport
close all; 
rng('default');

% line is fixed and data simulated above
from_line1; to_line1; mc_energy1; mc_condexp_target1;
from_line2; to_line2; mc_energy2; mc_condexp_target2;

% set up pde domain
dx = 0.1; 
dy = dx;    % spatial step size
ng = 2;             % number of ghost cells

% left right boundaries
x_min = 0; x_max = 3;
nx = ceil((x_max-x_min)/dx);
% top bottom boundaries
y_min = 0; y_max = 6;
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
tmp=mvksdensity([mc_energy1(:,1) mc_energy2(:,1)], [Xg(:) Yg(:)]);
tmp=reshape(tmp,[nx+4,ny+4]);
p(:,:,1)=tmp;
% visualize
figure(1);
surf(Xg,Yg,p(:,:,1)); view([90,90,90]);
title("IC"); xlabel("x"); ylabel("y"); zlabel("p(x,y,0)");

%% Time loop

for nn = 2:nt
    disp(nn)
    % estimate coefficients on cell edges
    x_data = squeeze(mc_energy1(:,nn));
    y_data = squeeze(mc_energy2(:,nn));
    % d/dx coefficient
    response_x_data = squeeze(mc_condexp_target1(:,nn));
    % mode for coefficient regression
    mode = "lin";       % linear regression
    coeff1 = get_coeff2d(x_data,y_data,response_x_data,xpts_e,ypts_e,mode);
    % d/dy coefficient
    response_y_data = squeeze(mc_condexp_target2(:,nn));
    coeff2 = get_coeff2d(x_data,y_data,response_y_data,xpts_e,ypts_e,mode);

    % zero out coefficients outside of main support
    coeff1 = clip_coeff(coeff1,Xge,Yge,x_data,y_data);
    coeff2 = clip_coeff(coeff2,Xge,Yge,x_data,y_data);

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
    if max(max(abs(p(:,:,nn))))>1e5
        error('PDE blows up');
    end
    if any(any(isnan(p(:,:,nn))))
        error('PDE has NaN values');
    end

    % visualize
    fig=figure(1);
    fig.Position = [100 500 1600 400];
    subplot(1,3,1);
    % plot predicted (RO-PDF)
    surf(p(3:end-2,3:end-2,nn));
    view([90,90,90]); 
    subplot(1,3,2);
    % plot exact
    %surf(p(3:end-2,3:end-2,nn));
    %view([90,90,90]); 
    subplot(1,3,3);
    % plot error
    %surf(abs(p(3:end-2,3:end-2,nn)-pkde(3:end-2,3:end-2,nn)));
    %view([90,90,90]); 
end

%% Helper subroutines
function pnext = transport(p1,coeff1,coeff2,dx,dy,dt)
    % Corner transport scheme to take a time step of dt for the 2d 
    % advection equation of form:
    %   p_t + ( c1(x, y) * p )_x + (c2(x, y) * p )_y = 0
    %
    % Ref: LeVeque sect 20.9

    pnext = p1;
    % get size of cell centers mesh
    [nx,ny] = size(p1(3:end-2,3:end-2));
    
    % positive and negative parts
    up = max(coeff1,0);
    um = min(coeff1,0);
    vp = max(coeff2,0);
    vm = min(coeff2,0);
    
    % left-right fluxes, i.e. x-direction

    % Apdq(i-1/2,j)
    Apdq = (up(3:end-2,3:end-2) + um(4:end-1,3:end-2)).*p1(3:end-2,3:end-2)...
            - (up(3:end-2,3:end-2).*p1(2:end-3,3:end-2) ...
            + um(3:end-2,3:end-2).*p1(3:end-2,3:end-2));
    % Amdq(i+1/2,j)
    Amdq = (up(4:end-1,3:end-2).*p1(3:end-2,3:end-2) ...
            + um(4:end-1,3:end-2).*p1(4:end-1,3:end-2))...
            - (up(3:end-2,3:end-2) + um(4:end-1,3:end-2)).*p1(3:end-2,3:end-2); 

    % up-down fluxes, i.e. y-direction
    Bpdq = (vp(3:end-2,3:end-2) + vm(3:end-2,4:end-1)).*p1(3:end-2,3:end-2)...
            - (vp(3:end-2,3:end-2).*p1(3:end-2,2:end-3) ...
            + vm(3:end-2,3:end-2).*p1(3:end-2,3:end-2));

    Bmdq = (vp(3:end-2,4:end-1).*p1(3:end-2,3:end-2) ...
            + vm(3:end-2,4:end-1).*p1(3:end-2,4:end-1))...
            - (vp(3:end-2,3:end-2) + vm(3:end-2,4:end-1)).*p1(3:end-2,3:end-2);
    
    
    % x-direction
  
    xWp = p1(4:end-1,2:end-1) - p1(3:end-2,2:end-1);
    % xWp = p1(4:end-1,3:end-2) - p1(3:end-2,3:end-2);
    

    xWm = p1(3:end-2,2:end-1) - p1(2:end-3,2:end-1);
    % xWm = p1(3:end-2,3:end-2) - p1(2:end-3,3:end-2);
    
    % y-direction
    yWp = p1(2:end-1,4:end-1) - p1(2:end-1,3:end-2);
    % yWp = p1(3:end-2,4:end-1) - p1(3:end-2,3:end-2);
    
    yWm = p1(2:end-1,3:end-2) - p1(2:end-1,2:end-3);
    % yWm = p1(3:end-2,3:end-2) - p1(3:end-2,2:end-3);
    
    
    % allocate thetas;
    
    % for Theta at i+1/2, j+1/2
    xThp = zeros(nx+4,ny+4); yThp = zeros(nx+4,ny+4);
    % for Theta at i-1/2, j-1/2
    xThm = zeros(nx+4,ny+4); yThm = zeros(nx+4,ny+4);
    
    % 1. Thetas for x direction
    % for theta x, loop over rows of u and find
    % > 0 and <= 0 entries, and fill in Theta 
    % correspondingly
    
    for r = 3:nx-2 % i-1/2,j => xThm 
       u_row = coeff1(r,:);
       % I = i-1
       positive_entries = find(u_row > 0);
       % I = i+1
       negative_entries = find(u_row <= 0);
       
       % these entries would correspond to theta entries...
       % note u is defined at i-1/2,j ...
       for idx = 1:length(positive_entries)
           % sect. (6.61) and (9.74)
           % W(I-1/2,j)/W(i-1/2,j) = W(i-3/2,j)/W(i-1/2,j)
           % W(i-3/2,j) = Q(i-1,j) - Q(i-2,j)
           c = positive_entries(idx);
           xThm(r,c) = (p1(r-1,c) - p1(r-2,c))/(p1(r,c) - p1(r-1,c));
       end
       for idx = 1:length(negative_entries)
           % W(I-1/2,j)/W(i-1/2,j) = W(i+1/2,j)/W(i-1/2,j)
           % W(i+1/2,j) = Q(i+1,j) - Q(i,j)
           c = negative_entries(idx);
           xThm(r,c) = (p1(r+1,c) - p1(r,c))/(p1(r,c) - p1(r-1,c));
       end
    end
    
    for r = 4:nx-1 % i+1/2,j => xThp
        u_row = coeff1(r,:);
        % I = i-1
        positive_entries = find(u_row > 0);
        % I = i+1
        negative_entries = find(u_row <= 0);
        
        % same logic, but for right cell edge thetas...
        for idx = 1:length(positive_entries)
            % W(I+1/2,j)/W(i+1/2,j) = W(i-1/2,j)/W(i+1/2,j)
            % W(i+1/2,j) = Q(i+1,j) - Q(i,j)
            c = positive_entries(idx);
            xThp(r,c) = (p1(r,c) - p1(r-1,c)) / (p1(r+1,c) - p1(r,c));
        end
        
        for idx = 1:length(negative_entries)
           % W(I+1/2,j)/W(i+1/2,j) = W(i+3/2,j)/W(i+1/2,j)
           % W(i+3/2,j) = Q(i+2,j) - Q(i+1,j)
           c = negative_entries(idx);
           xThp(r,c) = (p1(r+2,c) - p1(r+1,c)) / (p1(r+1,c) - p1(r,c));
        end
        
    end
    
    % 2. Thetas for y direction, similar logic
    
    for c = 3:ny-2 % i,j-1/2 => yThm
        v_col = coeff2(:,c);
        
        % I = i-1
        positive_entries = find(v_col > 0);
        % I = i+1
        negative_entries = find(v_col <= 0);
        
        for idx = 1:length(positive_entries)
            % W(i,J-1/2)/W(i,j-1/2) = W(i,j-3/2)/W(i,j-1/2)
            % W(i,j-3/2) = Q(i,j-1) - Q(i,j-2)
            r = positive_entries(idx);
            yThm(r,c) = (p1(r,c-1) - p1(r,c-2))/(p1(r,c) - p1(r,c-1));
        end
        
        for idx = 1:length(negative_entries)
            % W(i,J-1/2)/W(i,j-1/2) = W(i,j+1/2)/W(i,j-1/2)
            % W(i,j+1/2) = Q(i,j+1) - Q(i,j)
            r = negative_entries(idx);
            yThm(r,c) =  (p1(r,c+1) - p1(r,c))/(p1(r,c) - p1(r,c-1));
        end
    end
    
    for c = 4:ny-1 % i,j+1/2 => yThp
        v_col = coeff2(:,c);
        
        % I = i-1
        positive_entries = find(v_col > 0);
        % I = i+1
        negative_entries = find(v_col <= 0);
        
        for idx = 1:length(positive_entries)
           r = positive_entries(idx);
           % W(i,J+1/2)/W(i,j+1/2) = W(i,j-1/2)/W(i,j+1/2)
           % Q(i,j)-Q(i,j-1) / Q(i,j+1) - Q(i,j)
           yThp(r,c) = (p1(r,c) - p1(r,c-1)) / (p1(r,c+1) - p1(r,c));
        end
        
        for idx = 1:length(negative_entries)
           r = negative_entries(idx);
           % W(i,J+1/2)/W(i,j+1/2) = W(i,j+3/2)/W(i,j+1/2)
           % Q(i,j+2)-Q(i,j+1) / Q(i,j+1)-Q(i,j)
           yThp(r,c) = (p1(r,c+2) - p1(r,c+1)) / (p1(r,c+1) - p1(r,c));
        end
    end
   
    % 3. all thetas are filled in, do limiters (Van Leer)
    xPhim = max(max(0,min(1,2*xThm)),min(2,xThm));
    xPhip = max(max(0,min(1,2*xThp)),min(2,xThp));
    yPhim = max(max(0,min(1,2*yThm)),min(2,yThm));
    yPhip = max(max(0,min(1,2*yThp)),min(2,yThp));
    
    % modify waves 
    xWp = xPhip(3:end-2,2:end-1).*xWp;
    xWm = xPhim(3:end-2,2:end-1).*xWm;
    yWp = yPhip(2:end-1,3:end-2).*yWp;
    yWm = yPhim(2:end-1,3:end-2).*yWm;
    
    % second-order corrections
    % y-direction
    %(i,j+1/2)
    Gp = 0.5*abs(coeff2(2:end-1,4:end-1)).*(1-(dt/dx)* ...
        abs(coeff2(2:end-1,4:end-1))).*yWp;
    %(i,j-1/2)
    Gm = 0.5*abs(coeff2(2:end-1,3:end-2)).*(1-(dt/dx)* ...
        abs(coeff2(2:end-1,3:end-2))).*yWm;
    
    % x-direction
    %(i+1/2,j)
    Fp = 0.5*abs(coeff1(4:end-1,2:end-1)).*(1-(dt/dx)* ...
        abs(coeff1(4:end-1,2:end-1))).*xWp;
    %(i-1/2,j)
    Fm = 0.5*abs(coeff1(3:end-2,2:end-1)).*(1-(dt/dx)* ...
        abs(coeff1(3:end-2,2:end-1))).*xWm;

    % update solution to the next time step
    pnext(3:end-2,3:end-2) = p1(3:end-2,3:end-2) - dt*((Apdq + Amdq)/dx ...
        + (Bpdq + Bmdq)/dy + (Fp(:,2:end-1)-Fm(:,2:end-1))/dx ...
        + (Gp(2:end-1,:)-Gm(2:end-1,:))/dy);
end


% ------------------------------------------------------------------------
function c = get_coeff2d(x_data,y_data,z_data,xgrid,ygrid,mode)
    % Estimate coefficient using [x, y] => z data:
    % Exact solution is E[z | X=x, Y=y], and evalute 
    % result on meshgrid formed by (xgrid, ygrid) representing 
    % cell edges. `mode` indicates the parameteric or nonparameteric 
    % model used for regression.
    assert((numel(x_data)==numel(y_data))&(numel(y_data)==numel(z_data)));
    xx=x_data(:); yy=y_data(:); zz=z_data(:);
    % number of samples
    ns = numel(xx);
    % number of grid points in each dimension
    nx = length(xgrid); ny = length(ygrid);
    dx = xgrid(2)-xgrid(1); dy = ygrid(2)-ygrid(1);
    c = zeros(nx,ny);
    if mode == "lin"
        % linear regression with bias
        X=[ones(ns,1) xx yy];
        coeffs = (X'*X)\(X'*zz);
        % evaluate coefficients at each point in meshgrid
        for i = 1:nx
            for j = 1:ny
                c(i,j) = coeffs(2)*xgrid(i)+coeffs(3)*ygrid(j)+coeffs(1);
            end
        end
    else
        error("not implemented! ");
    end
    if any(any(isnan(c)))
        error("NaN encountered. ");
    end
end

% ------------------------------------------------------------------------
function c = clip_coeff(coeff,Xge,Yge,x_data,y_data)
    % determine a support based on observations data, and 
    % set all coefficients outside of that support to be 0.0
    c = coeff;
    xmin = min(x_data); xmax = max(x_data);
    ymin = min(y_data); ymax = max(y_data);
    idx = (Xge<xmin)&(Xge>xmax)&(Yge<ymin)&(Yge>ymax);
    c(idx) = 0.0;
end


