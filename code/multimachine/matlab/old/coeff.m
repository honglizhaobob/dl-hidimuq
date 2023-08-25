%% Coefficients via Smoothing splines

% Learning coefficients: need to run qoi_setup.m prior

% Direct from Fokker-Planck 
% df/dt + d/dy[ ((1-<x^2|y>)*y - <x|y> +sqrt(D)*<z|y>) f] = 0
% Cond. Expectations on gridE and midpoint in time

% Recall:
% idt2  = [1, (0.5*dt1/dt):(dt1/dt):nt];
% t2    = t(idt2);

clear; close all
addpath('./utils')
addpath('./Johnson_Curve')
% load('qoi_setup.mat')
load('qoi_setup2.mat')

coeff1 = zeros(nx+1,nt1,nmcro);   % <x^2|y>
coeff2 = zeros(nx+1,nt1,nmcro);   % <x|y>
coeff3 = zeros(nx+1,nt1,nmcro);   % <z|y>
coeff4 = zeros(nx+1,nt1,nmcro);   % < (1-x^2)*y - x +sqrt(D)*z | y >

imc0 = zeros(nt1,nmcro);
imc1 = zeros(nt1,nmcro);
p1 = zeros(nt1,nmcro);
p2 = zeros(nt1,nmcro);
p3 = zeros(nt1,nmcro);
p4 = zeros(nt1,nmcro);

% If another language more suitable to stats is used, 
% consider using quantile-parameterized distributions instead of the 
% transformations used below.

for j = 1:nmcro
     
    mcro = mcro_vec(j);
    [qq, iqs] = sort(qoi(1:mcro,:),1);
    r2 = x1(1:mcro,:);
    r3 = z(1:mcro,:);

    for i = 2:nt1
        rng('default')

        imc0(i,j) = find(gridE<=qq(1,i),1,'last');
        imc1(i,j) = find(gridE>=qq(mcro,i),1,'first');

        % Sorted by predictor data
        qs = qq(:,i);           % sorted predictor data
        r2s = r2(iqs(:,i),i);   % response data 2: x1    
        r1s = r2s.^2;           % response data 1: x1^2
        r3s = r3(iqs(:,i),i);   % response data 3: z
        r4s = (1-r1s).*qs - r2s + sqrt(D)*r3s;   % response data 4: whole thing
        
        % Transform predictor to approximately uniform[0,1] data: F_y(y(t);t)
        qt1 = interp1(gridE(imc0(i,j):imc1(i,j)), ...
                      squeeze(FqoiE(imc0(i,j):imc1(i,j),i,j)),qs,'pchip');
        % Transform uniform to Gaussian via inverse CDF
        qt2 = icdf('Normal',qt1,0,1); 
        mqt2 = mean(qt2);   sqt2 = std(qt2);
        qt3 = (qt2 - mqt2)/sqt2;

        % Box-Cox transformation of response data 1
        % r1t1 = (r1s.^lam - 1)/lam  if lam~=0
        % r1t1 = log(r1s)            if lam==0
        [r1t1, lam] = boxcox(r1s);
        mr1t1 = mean(r1t1);    sr1t1 = std(r1t1);
        r1t2  = (r1t1 - mr1t1)/sr1t1;   % standardize
        ifault1 = zeros(size(r1t1));
        mcro1 = mcro;

%         % Fit Johnson distribution to r1s
%         r1t1_struct = f_johnson_fit(r1s,'Q',0);
%         % Transform to standard normal
%         [r1t1,ifault1] = f_johnson_y2z(r1s,r1t1_struct.coef,r1t1_struct.type);
%         r1t1 = r1t1(ifault1==0);
%         mcro1 = length(r1t1);        

        % Fit Johnson distribution to r2s
        r2t1_struct = f_johnson_fit(r2s,'Q',0);
        % Transform to standard normal
        [r2t1,ifault2] = f_johnson_y2z(r2s,r2t1_struct.coef,r2t1_struct.type);
        r2t1 = r2t1(ifault2==0);
        mcro2 = length(r2t1);

        % Only standardize r3
        mr3s = mean(r3s);   sr3s = std(r3s);
        r3t1 = (r3s - mr3s)/sr3s;

        mr4s = mean(r4s);   sr4s = std(r4s);
        r4t1 = (r4s - mr4s)/sr4s;
        % Fit Johnson distribution to r4s
        r4t2_struct = f_johnson_fit(r4t1,'Q',0);
        if r4t2_struct.coef(2)<=0 || r4t2_struct.coef(4)<=0
            r4t2 = r4t1;
            ifault4 = zeros(size(r4t1));
            mcro4 = mcro;
        else
            % Transform to standard normal
            [r4t2,ifault4] = f_johnson_y2z(r4t1,r4t2_struct.coef,r4t2_struct.type);
            r4t2 = r4t2(ifault4==0);
            mcro4 = length(r4t2);
        end

        % Smoothing parameter: 
        % Bandwidths optimal for Gaussian local linear regression
        bwq1  = median(abs(qt3(ifault1==0)-median(qt3(ifault1==0))))/0.6745*(4/3/mcro1)^0.2;
        bwq2  = median(abs(qt3(ifault2==0)-median(qt3(ifault2==0))))/0.6745*(4/3/mcro2)^0.2;
        bwq3  = median(abs(qt3-median(qt3)))/0.6745*(4/3/mcro)^0.2;
        bwq4  = median(abs(qt3(ifault4==0)-median(qt3(ifault4==0))))/0.6745*(4/3/mcro4)^0.2;


        bwr1 = median(abs(r1t2-median(r1t2)))/0.6745*(4/3/mcro1)^0.2;
        bwr2 = median(abs(r2t1-median(r2t1)))/0.6745*(4/3/mcro2)^0.2;
        bwr3 = median(abs(r3t1-median(r3t1)))/0.6745*(4/3/mcro)^0.2;
        bwr4 = median(abs(r4t2-median(r4t2)))/0.6745*(4/3/mcro4)^0.2;

        p1(i,j) = 1/(1 + sqrt(bwq1*bwr1));
        p2(i,j) = 1/(1 + sqrt(bwq2*bwr2));
        p3(i,j) = 1/(1 + sqrt(bwq3*bwr3));
        p4(i,j) = 1/(1 + sqrt(bwq4*bwr4));

        % Now iteratively choose bisquare weights for smoothing spline
        % Only 1 iteration for now
%         w1 = csaps_weights(qt3(ifault1==0),r1t1,p1(i,j),1,1e-3);
%         w2 = csaps_weights(qt3(ifault2==0),r2t1,p2(i,j),1,1e-3);
%         w3 = csaps_weights(qt3,r3t1,p3(i,j),1,1e-3);
%         w4 = csaps_weights(qt3(ifault4==0),r4t1,p4(i,j),1,1e-3);
        w1 = ones(size(r1t1));
        w2 = ones(size(r2t1));
        w3 = ones(size(r3t1));
        w4 = ones(size(r4t2));

        % Transformed phase space grid for qs
        gEt1 = icdf('Normal',squeeze(FqoiE(imc0(i,j):imc1(i,j),i,j)),0,1);
        gEt2 = (gEt1 - mqt2)/sqt2;

        % Fit smoothing splines with 2nd-order extrap
        spline1 = fnval(fnxtr(csaps(qt3(ifault1==0),r1t2,p1(i,j),[],w1), 1), gEt2);
        spline2 = fnval(fnxtr(csaps(qt3(ifault2==0),r2t1,p2(i,j),[],w2), 1), gEt2);
        spline3 = fnval(fnxtr(csaps(qt3,r3t1,p3(i,j),[],w3), 1), gEt2);
        spline4 = fnval(fnxtr(csaps(qt3(ifault4==0),r4t2,p4(i,j),[],w4), 1), gEt2);

        % Take inverse transforms for predictions
        if lam == 0
            spline1 = exp(spline1*sr1t1 + mr1t1);
        else
            spline1 = ((spline1*sr1t1 + mr1t1)*lam + 1).^(1/lam);
            spline1 = max(real(spline1),0);
        end
%         spline1 = f_johnson_z2y(spline1,r1t1_struct.coef,r1t1_struct.type);
        spline2 = f_johnson_z2y(spline2,r2t1_struct.coef,r2t1_struct.type);
        spline3 = spline3*sr3s + mr3s;
        if r4t2_struct.coef(2)<=0 || r4t2_struct.coef(4)<=0
            spline4 = spline4*sr4s + mr4s;
        else
            spline4 = f_johnson_z2y(spline4,r4t2_struct.coef,r4t2_struct.type)...
                      *sr4s + mr4s;
        end

        coeff1(imc0(i,j):imc1(i,j),i,j) = spline1;
        coeff2(imc0(i,j):imc1(i,j),i,j) = spline2;
        coeff3(imc0(i,j):imc1(i,j),i,j) = spline3;
        coeff4(imc0(i,j):imc1(i,j),i,j) = spline4;


%     figure(1)
%     plot(qs,r1s,'ok','markersize',1)
%     hold on
%     plot(gridE(imc0(i,j):imc1(i,j)),spline1,'linewidth',1.5); hold off
%     shg
% 
%     figure(2)
%     plot(qs,r2s,'ok','markersize',1)
%     hold on
%     plot(gridE(imc0(i,j):imc1(i,j)),spline2,'linewidth',1.5); hold off
%     shg

        disp([i,j])
    end
end


% fails: (659,3), (784,3), ~(350,4), (435,4), 489,760,786


%% Check for failures
c1Fail = [];
c2Fail = [];
c3Fail = [];
c4Fail = [];

for j = 1:nmcro
    for i = 2:nt1
           c1 = coeff1(imc0(i,j):imc1(i,j),i,j);
           c2 = coeff2(imc0(i,j):imc1(i,j),i,j);
           c3 = coeff3(imc0(i,j):imc1(i,j),i,j);
           c4 = coeff3(imc0(i,j):imc1(i,j),i,j);

        if any(isnan(c1))||any(abs(c1)==Inf)||~isreal(c1)
            c1Fail = [c1Fail; [i j]];
        end
        if any(isnan(c2))||any(abs(c2)==Inf)||~isreal(c2)
            c2Fail = [c2Fail; [i j]];
        end
        if any(isnan(c3))||any(abs(c3)==Inf)||~isreal(c3)
            c3Fail = [c3Fail; [i j]];
        end
        if any(isnan(c4))||any(abs(c4)==Inf)||~isreal(c4)
            c4Fail = [c4Fail; [i j]];
        end
    end
end

% All are empty

%% Plotting for inspection

j = 3;

mcro = mcro_vec(j);
[qq, iqs] = sort(qoi(1:mcro,:),1);
r2 = x1(1:mcro,:);
r3 = z(1:mcro,:);

for i = 2:nt1

    % Sorted by predictor data
    qs = qq(:,i);           % sorted predictor data
    r2s = r2(iqs(:,i),i);   % response data 2: x1    
    r1s = r2s.^2;           % response data 1: x1^2
    r3s = r3(iqs(:,i),i);   % response data 3: z

    c1 = coeff1(imc0(i,j):imc1(i,j),i,j);
    c2 = coeff2(imc0(i,j):imc1(i,j),i,j);
    c3 = coeff3(imc0(i,j):imc1(i,j),i,j);

    figure(1)
    plot(qs,r1s,'ok','markersize',1)
    hold on
    plot(gridE(imc0(i,j):imc1(i,j)),c1,'linewidth',1.5); hold off
    shg
    
    figure(2)
    plot(qs,r2s,'ok','markersize',1)
    hold on
    plot(gridE(imc0(i,j):imc1(i,j)),c2,'linewidth',1.5); hold off
    shg
    
    figure(3)
    plot(qs,r3s,'ok','markersize',1)
    hold on
    plot(gridE(imc0(i,j):imc1(i,j)),c3,'linewidth',1.5); hold off
    shg

    disp(i)

end


%% Constuct advect and Save
advect  = zeros(nx+1,nt1,nmcro);
advect_sm = zeros(nx+1,nt1,nmcro);
coeff4_sm = zeros(nx+1,nt1,nmcro);

for j = 1:nmcro
    for i = 2:nt1
%         disp(i)

        ig0 = max(imc0(i,j)-10,1);
        ig1 = min(imc1(i,j)+10,nx+1);

        advect(imc0(i,j):imc1(i,j),i,j) = ( (1-coeff1(imc0(i,j):imc1(i,j),i,j))...
               .*(gridE(imc0(i,j):imc1(i,j))*(qoiMax-qoiMin) + qoiMin) ...
               - coeff2(imc0(i,j):imc1(i,j),i,j) ...
               + sqrt(D)*coeff3(imc0(i,j):imc1(i,j),i,j) ) ...
               / (qoiMax-qoiMin);

        % Smooth advect at support boundaries
        if imc0(i,j)>ig0 && imc1(i,j)<ig1
    
            advect(:,i,j) = interp1([gridE(1:ig0);gridE(imc0(i,j):imc1(i,j));gridE(ig1:(nx+1))],...
                      [zeros(ig0,1);advect(imc0(i,j):imc1(i,j),i,j);zeros(nx+2-ig1,1)],...
                      gridE,'makima');

            coeff4(:,i,j) = interp1([gridE(1:ig0);gridE(imc0(i,j):imc1(i,j));gridE(ig1:(nx+1))],...
                      [zeros(ig0,1);coeff4(imc0(i,j):imc1(i,j),i,j);zeros(nx+2-ig1,1)],...
                      gridE,'makima');
        elseif imc0(i,j)>ig0
            advect(:,i,j) = interp1([gridE(1:ig0);gridE(imc0(i,j):imc1(i,j))],...
                      [zeros(ig0,1);advect(imc0(i,j):imc1(i,j),i,j)],...
                      gridE,'makima');
            coeff4(:,i,j) = interp1([gridE(1:ig0);gridE(imc0(i,j):imc1(i,j))],...
                      [zeros(ig0,1);coeff4(imc0(i,j):imc1(i,j),i,j)],...
                      gridE,'makima');
        elseif imc1(i,j)<ig1
            advect(:,i,j) = interp1([gridE(imc0(i,j):imc1(i,j));gridE(ig1:(nx+1))],...
                      [advect(imc0(i,j):imc1(i,j),i,j);zeros(nx+2-ig1,1)],...
                      gridE,'makima');
            coeff4(:,i,j) = interp1([gridE(imc0(i,j):imc1(i,j));gridE(ig1:(nx+1))],...
                      [coeff4(imc0(i,j):imc1(i,j),i,j);zeros(nx+2-ig1,1)],...
                      gridE,'makima');
        end

        advect_sm(:,i,j) = smoothdata(squeeze(advect(:,i,j)),'gaussian',10)';
        coeff4_sm(:,i,j) = smoothdata(squeeze(coeff4(:,i,j)),'gaussian',10)';
    end
end


% save('coeffs.mat','coeff1','coeff2','coeff3','advect','advect_sm',...
%                   'coeff4','coeff4_sm','imc0','imc1','p1','p2','p3','-v7.3')

save('coeffs2.mat','coeff1','coeff2','coeff3','advect','advect_sm',...
                  'coeff4','coeff4_sm','imc0','imc1','p1','p2','p3','-v7.3')


%%
% pdat  = (x2(1:mcro, idt2)-qoiMin)./(qoiMax-qoiMin);
% rdat1 = x1(1:mcro, idt2).^2;
% rdat2 = x1(1:mcro, idt2);
% rdat3 = z(1:mcro, idt2);
% 
% pmcmin = min((x2(:, idt2)-qoiMin)./(qoiMax-qoiMin));
% pmcmax = max((x2(:, idt2)-qoiMin)./(qoiMax-qoiMin));
% r1mcmin = min(x1(:, idt2).^2);
% r1mcmax = max(x1(:, idt2).^2);
% r2mcmin = min(x1(:, idt2));
% r2mcmax = max(x1(:, idt2));
% r3mcmin = min(z(:, idt2));
% r3mcmax = max(z(:, idt2));
% 
% 
% for i = 2:nt1
%     idx0(i) = max(binarySearch(gridE, pmcmin(i)), 1);
%     idx1(i) = min(binarySearch(gridE, pmcmax(i))+1, nx+1);
% 
%     pred  = (pdat(:,i) - pmcmin(i))./(pmcmax(i) - pmcmin(i));
%     resp1 = (rdat1(:,i) - r1mcmin(i))./(r1mcmax(i) - r1mcmin(i));
%     resp2 = (rdat2(:,i) - r2mcmin(i))./(r2mcmax(i) - r2mcmin(i));
%     resp3 = (rdat3(:,i) - r3mcmin(i))./(r3mcmax(i) - r3mcmin(i));
% 
%     bwp  = median(abs(pred-median(pred)))/0.6745*(4/3/mcro)^0.2;
%     bwr1 = median(abs(resp1-median(resp1)))/0.6745*(4/3/mcro)^0.2;
%     bwr2 = median(abs(resp2-median(resp2)))/0.6745*(4/3/mcro)^0.2;
%     bwr3 = median(abs(resp3-median(resp3)))/0.6745*(4/3/mcro)^0.2;
%         
%     bw1(i) = 1-sqrt(bwp*bwr1);
%     bw2(i) = 1-sqrt(bwp*bwr2);
%     bw3(i) = 1-sqrt(bwp*bwr3);
% end
% 
% % Adjusted smoothing params: manually tuned
% bw1([2,3]) = .92;
% bw1([250,577,914,935]) = .95;
% bw1([232,274,276,547,555,591,595,599,908,915,922]) = .96;
% bw1([177,222,224,227,311,531,534,541,543,557,596,602,885,886,...
%      891,893,894,927,932]) = .97;
% bw1([147,149,166,171,180,207,213,277,278,289,294,297,343,423,...
%      428,441,445,587,614,616,618,623:626,745,748,790,827,828,...
%      961,990]) = .98;
% bw1([308,313,316,321,330,331,340,348,350,355,378:387,390:392, ...
%      407,491,506,515,519,628,631,633:638,642:649,656,664,667,...
%      687,690,693:695,699,724,727,740,744,747,772,773,785,788,...
%      794:796,802,811,820,838,859,862,863,866,870,933,951:960,...
%      963:984,986:989,991,993:995]) = .99;
% bw1([287,304,314,320,325,326,356,367,369:371,380,381,388:389,...
%      394,431,639,640,650:654,657,662,665,668,672,676,678,681,...
%      683,684,701,705:711,717,720,832,937,985]) = .999;
% 
% 
% bw2([252,922]) = .94;
% bw2([249,897,899]) = .95;
% bw2([13,271,863,866,904,914]) = .96;
% bw2([32,278,279,857:858,875,885,888,893,927]) = .97;
% bw2([68,156,167,176,187,297,432,434,541,543,549,778,781,783,...
%      785,806,832,843]) = .98;
% bw2([289,308,329:331,339,340,351,369,378,413,418,423,441,458,...
%      490,495,507:513,519:520,561,586,595,601,613:616,634,644,...
%      658,667,669,692,695,713,724,740,764,773,775,788,851,859,...
%      870,915,933,935,937,951,956,958,961,969]) = .99;
% bw2([320,333,344,359,371,373,380,381,383,404,407,431,651,662,...
%      663,665,677,681,709,720,721,747,975,985,989,993:1001]) = .999;
% 
% 
% bw3([13,19,53,122,144,156,207,211,227,245,251,587,922:923]) = .97;
% bw3(937) = .973;
% bw3([180,263,274,277:278,283,513,515,549,555,562,566,585,599,...
%      601:602,824,827,832,838,841,908,914,945,948]) = .98;
% bw3([252,287,289,293:294,303,305,308,316,328:331,340,349,358,...
%      375,391,411,496,509,519,530,603,613,615,616,619,634,653,...
%      664,678,688,732,776,781,788,951,975,993,1000]) = .99;
% bw3([320,326,333,344,352,356,367,371,380,381,391,665,705,710,...
%      713,720,727,772,773,995]) = .999;
% 
% coeff1 = zeros(nx+1,nt1); 
% coeff2 = zeros(nx+1,nt1);
% coeff3 = zeros(nx+1,nt1);
% advect = zeros(nx+1,nt1);
% 
% coeff1(:,1) = 1+sd^2;    % at t=0 -> = <x^2> = var(x) + <x>^2
% coeff2(:,1) = 1;         % at t=0 -> = <x> = 1
% % coeff3(:,1) = 0;       % at t=0 -> = <z> = 0
% 
% for i = 2:nt1
% 
%     ig0 = max(idx0(i)-7,1);
%     ig1 = min(idx1(i)+7,nx+1);
% 
%     pred  = (pdat(:,i) - pmcmin(i))./(pmcmax(i) - pmcmin(i));
%     resp1 = (rdat1(:,i) - r1mcmin(i))./(r1mcmax(i) - r1mcmin(i));
%     resp2 = (rdat2(:,i) - r2mcmin(i))./(r2mcmax(i) - r2mcmin(i));
%     resp3 = (rdat3(:,i) - r3mcmin(i))./(r3mcmax(i) - r3mcmin(i));
% 
%     coeff = 1 - (fnval(fnxtr(csaps(pred, resp1,  bw1(i)), 2), ...
%                   (gridE(idx0(i):idx1(i))-pmcmin(i))/(pmcmax(i) - pmcmin(i)))...
%           * (r1mcmax(i) - r1mcmin(i)) + r1mcmin(i));
%     coeff(coeff>1) = 1;
%     coeff1(idx0(i):idx1(i),i) = coeff;
%     
%     coeff2(idx0(i):idx1(i),i) = fnval(fnxtr(csaps(pred, resp2,  bw2(i)),2), ...
%                                 (gridE(idx0(i):idx1(i))-pmcmin(i))/(pmcmax(i) - pmcmin(i)))...
%                                 * (r2mcmax(i) - r2mcmin(i)) + r2mcmin(i);
% 
%     coeff3(idx0(i):idx1(i),i) = fnval(fnxtr(csaps(pred, resp3,  bw3(i)),2), ...
%                                 (gridE(idx0(i):idx1(i))-pmcmin(i))/(pmcmax(i) - pmcmin(i)))...
%                                 * (r3mcmax(i) - r3mcmin(i)) + r3mcmin(i);
% 
%     advect(idx0(i):idx1(i),i) = ( coeff1(idx0(i):idx1(i),i)...
%                                  .*(gridE(idx0(i):idx1(i))*(qoiMax-qoiMin) + qoiMin) ...
%                                 - coeff2(idx0(i):idx1(i),i) ...
%                                 + sqrt(D)*coeff3(idx0(i):idx1(i),i) ) ...
%                                 / (qoiMax-qoiMin);
% 
%     advect(:,i) = interp1([gridE(1:ig0);gridE(idx0(i):idx1(i));gridE(ig1:(nx+1))],...
%                   [zeros(ig0,1);advect(idx0(i):idx1(i),i);zeros(nx+2-ig1,1)],gridE,'makima');
% 
%     
% end
