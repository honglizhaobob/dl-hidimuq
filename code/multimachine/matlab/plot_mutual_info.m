% Plotting mutual information estimated from RO-PDF solution
clear; clc; rng("default");
%% 
% load computed solutions
case9 = load("./data/CASE9_2d_ROPDF_Sol.mat");
case30 = load("./data/CASE30_2d_KDE_Sol.mat");
case57 = load("./data/CASE57_2d_ROPDF_Sol.mat");
%%
tt = case9.tt;
nt = length(tt);
all_cases = {case9,case30,case57};
all_mutual_info = zeros(length(all_cases),nt);
for i = 1:length(all_cases)
    c = all_cases{i};
    dx = c.xpts(2)-c.xpts(1);
    dy = c.ypts(2)-c.ypts(1);
    for j = 1:nt
        j
        if i == 3
            all_mutual_info(i,j) = mutual_info(c.p2(3:end-2,3:end-2,j),dx,dy);
        elseif i == 2
            all_mutual_info(i,j) = mutual_info(c.p_kde(3:end-2,3:end-2,j),dx,dy);
        else
            all_mutual_info(i,j) = mutual_info(c.p(3:end-2,3:end-2,j),dx,dy);
        end
    end
end

%% Plot mutual information computed
fig = figure(1);
fig.Position = [500 500 900 800];
plot(tt,all_mutual_info(1,:),"LineWidth",3.5);
hold on;
plot(tt,all_mutual_info(2,:),":","LineWidth",3.5);
hold on;
plot(tt,all_mutual_info(3,:),"-.","LineWidth",3.5);

legend(["Case 9: Line 4-9,7-8","Case 30: Line 6-7,6-9", ...
    "Case 57: Line 35-36,36-40"], ...
    "FontSize",30,"FontName","Times New Roman","Location","northwest");
xlabel("Time","FontSize",30,"FontName","Times New Roman");
ylabel("Mutual Information","FontSize",30,"FontName","Times New Roman");
ax = gca;
ax.FontSize = 30;
box on;
ax = gca;
ax.LineWidth = 2;
% save figure
exportgraphics(fig,"./fig/Mutual_Info.png","Resolution",300);





