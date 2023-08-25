% Testing LOWESS local regression for 2d data set
% Hongli Zhao, 08/25/2023

%% Built-in dataset
clear; clc; rng("default");
load franke
f = fit([x y],z,'lowess');
plot(f,[x y],z);

%% IEEE Case 9 line energy
clear; clc; rng("default");

% load data
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
from_line1 = 4; to_line1 = 9;
from_line2 = 7; to_line2 = 8;

% create grid
dx = 0.02; 
dy = dx*4;    % spatial step size
ng = 2;             % number of ghost cells

% left right boundaries
x_min = 0; x_max = 1;
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

mcro = 1000;
% fit and plot 
for i = 1:nt

    % get data for this time step
    x1 = mc_energy1(1:mcro,i);
    x2 = mc_energy2(1:mcro,i);
    y1 = mc_condexp_target1(1:mcro,i);
    y2 = mc_condexp_target2(1:mcro,i);

    % fit with LOWESS
    f1 = fit([x1 x2],y1,'lowess');
    f2 = fit([x1 x2],y2,'lowess');

    % evaluate on regular grid
    f1pred = reshape(f1(Xge(:),Yge(:)),[nx+4 ny+4]);
    f2pred = reshape(f2(Xge(:),Yge(:)),[nx+4 ny+4]);

    % plot
    fg = figure(1); 
    fg.Position = [500,500,1200,600];
    subplot(1,2,1);
    scatter3(x1,x2,y1,2.0,"black"); hold on;
    surf(Xge,Yge,f1pred); hold off;
    
    subplot(1,2,2);
    scatter3(x1,x2,y2,2.0,"black"); hold on;
    surf(Xge,Yge,f2pred); hold off;
end














