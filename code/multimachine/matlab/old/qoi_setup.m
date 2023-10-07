%% QoI setup: Domain and MC KDE

% Must first run sde_mc.m to get SDE samples
% File computes and saves grids and KDE estimators

clear
load('sde_mc.mat')

addpath('./kde')
addpath('./utils')

rng('default')

mcro_vec = [5e2,1e3,3e3,5e3];    % No. of MC sample for RO-PDF

mcro_vec = sort(mcro_vec);
nmcro = length(mcro_vec);
assert(mcro_vec(end)<mc)

% Coarse time grid for PDEs
% dt1 = 1e-2;        % time step
dt1 = 2e-3;        % time step version 2
assert(dt <= dt1)
assert(mod(dt1/dt,2)==0)

t1  = 0:dt1:tf;
nt1 = length(t1);
idt = 1:(dt1/dt):nt;

% QoI
qoi = x2(:,idt);
assert(isequal(size(qoi), [mc,nt1]))

% Spatial grid for PDEs
% nx   = 200;       % Spatial cells
nx   = 1000;       % Spatial cells verion 2
pad  = 0.025;      % Percentage of data range to pad PDE domain 
ng   = 2;          % Number of ghost cells for PDE scheme: (=2 for lax-wen)

assert((pad > 0.0) && (pad < 1.0))
assert(ng == 2)

qoiMin = min(qoi,[],'all');  
qoiMax = max(qoi,[],'all');
qoiR   = qoiMax - qoiMin;
% Add in padding
qoiMin = qoiMin - qoiR*pad;
qoiMax = qoiMax + qoiR*pad;

% Transform domain to [0,1]: grid(x) = (x - qoiMin)/(qoiMax-qoiMin)
qoiMC = (qoi((mcro_vec(end)+1):mc,:)-qoiMin)./(qoiMax-qoiMin);
qoi   = (qoi(1:mcro_vec(end),:)-qoiMin)./(qoiMax-qoiMin);

dx    = 1/nx;                                  % Cell size
gridG = (dx*(0.5-ng) :dx: (1+dx*(ng-0.5)))';   % Grid w/ ghost cells
idx   = (ng+1):(nx+ng);                        % Indices on non-ghost cells
grid  = gridG(idx);                            % Grid cells
gridE = gridG((ng+1):(nx+ng+1)) - 0.5*dx;      % Grid edges for lax-wen 


fmc  = zeros(nx,nt1);
fqoi = zeros(nx,nt1,nmcro);
FqoiE = zeros(nx+1,nt1,nmcro);

% Benchmark MC KDE
parfor i = 1:nt1
    fmc(:,i) = akde1d(qoiMC(:,i), grid);
    fmc(:,i) = fmc(:,i)/trapz(grid, fmc(:,i));  % normalize
end
% MC KDE estimates w/ samples sizes from mcro_vec(j)
parfor j = 1:nmcro
    mcro = mcro_vec(j);
    for i = 1:nt1
        fqoi(:,i,j)  = akde1d(qoi(1:mcro,i), grid);
        fqoi(:,i,j)  = fqoi(:,i,j)/trapz(grid, squeeze(fqoi(:,i,j)));
        % CDF on grid edges --> use later to transform data
        FqoiE(:,i,j) = interp1(grid, cumtrapz(grid,squeeze(fqoi(:,i,j))),...
                               gridE,'pchip','extrap');
    end
end
FqoiE(1,:,:) = 0;     % Boundary Conditions for CDF
FqoiE(nx+1,:,:) = 1;

clear qoiMC

idt2  = [1, (0.5*dt1/dt):(dt1/dt):nt];  
t2    = t(idt2);

% Re-define states on midpoint in time and smaller samples for learning
x1 = x1(1:mcro_vec(nmcro), idt2);
x2 = x2(1:mcro_vec(nmcro), idt2);
z  = z(1:mcro_vec(nmcro), idt2);

% save('qoi_setup.mat','-v7.3')
save('qoi_setup2.mat','-v7.3')


