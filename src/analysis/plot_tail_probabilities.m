% Given computed RO-PDF solutions, plot surface plot of tail probabilities
clear; clc; rng("default");
%% Case 9
case9 = load("./data/CASE9_2d_ROPDF_Sol.mat");
% compute step sizes
dx = case9.xpts(2)-case9.xpts(1);
dy = case9.ypts(2)-case9.ypts(1);
dt = case9.tt(2)-case9.tt(1);
% time points t = 1.25, 2.5, 3.75, 5.00
idx = linspace(250,1000,4);
% meshgrid for plotting
[Yg,Xg]=meshgrid(case9.ypts,case9.xpts);
Yg = Yg(3:end-2,3:end-2);
Xg = Xg(3:end-2,3:end-2);
for ii = 1:length(idx)
    i = idx(ii);
    % compute tail probability
    pi = case9.p(3:end-2,3:end-2,i);
    [tailprob,~,~,~] = tail_prob2d(pi,dx,dy);
    fig = figure(1);
    fig.Position = [500 500 1000 900];
    subplot(2,2,ii);
    %imagesc(case9.xpts,case9.ypts,tailprob, ...
    %    "Interpolation",'bilinear');
    surf(Xg,Yg,tailprob,"EdgeAlpha",0.5);
    title(sprintf("$t=%.2f$", dt*i),"Interpreter","latex","FontSize",30);
    colormap jet; 
    % adjust y axis
    set(gca,"YDir","normal");
    % set fontsize for tick labels
    ax = gca;
    ax.FontSize = 30;
    box on;
    ax = gca;
    ax.LineWidth = 2;
    % adjust whitespace around subplots


end
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Line 4-9 Energy',"FontSize",30,'Position',[-0.08 0.5], ...
    "FontName","Times New Roman");
xlabel(han,'Line 7-8 Energy',"FontSize",30,'Position',[0.5 -0.05], ...
    "FontName","Times New Roman");
title(han,'Case 9',"FontSize",30,'Position',[0.5 1.02], ...
    "FontName","Times New Roman");
cb = colorbar("EastOutside","FontSize",20); clim([0 1]);
set(cb,'position',[0.93 0.3 .01 .5])
% save figure
exportgraphics(fig,"./fig/CASE9_ROPDF_TailProb.png","Resolution",300);
%% Case 30
case9 = load("./data/CASE30_2d_ROPDF_Sol.mat");
% compute step sizes
dx = case9.xpts(2)-case9.xpts(1);
dy = case9.ypts(2)-case9.ypts(1);
dt = case9.tt(2)-case9.tt(1);
% time points t = 1.25, 2.5, 3.75, 5.00
idx = linspace(250,1000,4);
% meshgrid for plotting
[Yg,Xg]=meshgrid(case9.ypts,case9.xpts);
Yg = Yg(3:end-2,3:end-2);
Xg = Xg(3:end-2,3:end-2);
for ii = 1:length(idx)
    i = idx(ii);
    % compute tail probability
    pi = case9.p(3:end-2,3:end-2,i);
    [tailprob,~,~,~] = tail_prob2d(pi,dx,dy);
    fig = figure(1);
    fig.Position = [500 500 1000 900];
    subplot(2,2,ii);
    %imagesc(case9.xpts,case9.ypts,tailprob, ...
    %    "Interpolation",'bilinear');
    surf(Xg,Yg,tailprob,"EdgeAlpha",0.5);
    title(sprintf("$t=%.2f$", dt*i),"Interpreter","latex","FontSize",30);
    colormap jet; 
    % adjust y axis
    set(gca,"YDir","normal");
    % set fontsize for tick labels
    ax = gca;
    ax.FontSize = 30;
    box on;
    ax = gca;
    ax.LineWidth = 2;
    % adjust whitespace around subplots


end
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Line 6-9 Energy',"FontSize",30,'Position',[-0.08 0.5], ...
    "FontName","Times New Roman");
xlabel(han,'Line 6-7 Energy',"FontSize",30,'Position',[0.5 -0.05], ...
    "FontName","Times New Roman");
title(han,'Case 30',"FontSize",30,'Position',[0.5 1.02], ...
    "FontName","Times New Roman");
cb = colorbar("EastOutside","FontSize",20); clim([0 1]);
set(cb,'position',[0.93 0.3 .01 .5])
% save figure
exportgraphics(fig,"./fig/CASE30_ROPDF_TailProb.png","Resolution",300);
%% Case 57
case9 = load("./data/CASE57_2d_ROPDF_Sol.mat");
% compute step sizes
dx = case9.xpts(2)-case9.xpts(1);
dy = case9.ypts(2)-case9.ypts(1);
dt = case9.tt(2)-case9.tt(1);
% time points t = 1.25, 2.5, 3.75, 5.00
idx = linspace(250,1000,4);
% meshgrid for plotting
[Yg,Xg]=meshgrid(case9.ypts,case9.xpts);
Yg = Yg(3:end-2,3:end-2);
Xg = Xg(3:end-2,3:end-2);
for ii = 1:length(idx)
    i = idx(ii);
    % compute tail probability
    pi = case9.p2(3:end-2,3:end-2,i);
    [tailprob,~,~,~] = tail_prob2d(pi,dx,dy);
    fig = figure(1);
    fig.Position = [500 500 1000 900];
    subplot(2,2,ii);
    %imagesc(case9.xpts,case9.ypts,tailprob, ...
    %    "Interpolation",'bilinear');
    surf(Xg,Yg,tailprob,"EdgeAlpha",0.5);
    title(sprintf("$t=%.2f$", dt*i),"Interpreter","latex","FontSize",30);
    colormap jet; 
    % adjust y axis
    set(gca,"YDir","normal");
    % set fontsize for tick labels
    ax = gca;
    ax.FontSize = 30;
    box on;
    ax = gca;
    ax.LineWidth = 2;
    % adjust whitespace around subplots


end
han=axes(fig,'visible','off'); 
han.Title.Visible='on';
han.XLabel.Visible='on';
han.YLabel.Visible='on';
ylabel(han,'Line 35-36 Energy',"FontSize",30,'Position',[-0.08 0.5], ...
    "FontName","Times New Roman");
xlabel(han,'Line 36-40 Energy',"FontSize",30,'Position',[0.5 -0.05], ...
    "FontName","Times New Roman");
title(han,'Case 57',"FontSize",30,'Position',[0.5 1.02], ...
    "FontName","Times New Roman");
cb = colorbar("EastOutside","FontSize",20); clim([0 1]);
set(cb,'position',[0.93 0.3 .01 .5])
% save figure
exportgraphics(fig,"./fig/CASE57_ROPDF_TailProb.png","Resolution",300);