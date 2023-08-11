% RO-PDF Methods

% close all
clear
addpath('./utils')

load('qoi_setup.mat')
load('coeffs.mat');

% load('qoi_setup2.mat')
% load('coeffs2.mat');

% Sparsity factor for nudging and splitting
nu_vec = [1,2,4,8,16];
nnu = length(nu_vec);

% Starting time (t1) index
nt0 = 2; 

% final time 2
tf2 = 6;
ntf = find(t1 == tf2);
ntf = max(nt0:max(nu_vec):ntf) + max(nu_vec);
assert(ntf<=nt1)

%% Solve PDE - no source

cfls = zeros(nmcro,1);
fpde = zeros(nx,ntf,nmcro);

for j = 1:nmcro

    cfls(j) = dx/max(abs(advect(:,3:nt1,j)),[],'all');

    fpde(:,1:nt0,j) = fmc(:,1:nt0);
    fg = zeros(nx+2*ng,1);

    for i = (nt0+1):ntf
        disp(i)
        fg(idx) = squeeze(fpde(:,i-1,j));
        fpde(:,i,j) = laxWen1d(fg, idx, nx, squeeze(advect(:,i,j)), dx, dt1);
    end

%     figure(j)
%     plot(grid,squeeze(fpde(:,ntf2,j)),grid,fmc(:,ntf2),'linewidth',2);shg
%     set(gca,'fontsize',22,'linewidth',2)
%     xlabel('X_2')
%     ylabel('f_{X_2}(X_2; t)')
%     title('Velocity PDF at t = 6')
%     legend('RO-PDF','MC')
%     ylim([0 14])
%     shg
end

% figure(3)
% plot(gridE(imc0(k,j):imc1(k,j)),squeeze(coeff2(imc0(k,j):imc1(k,j),k,j)))


%%  Kinetic defect: Splitting method

fpde_split = zeros(nx,ntf,nmcro,nnu);
src = zeros(nx,ntf,nmcro,nnu);

for k = 1:nnu
    nu = nu_vec(k);
    idt = [1:nt0,(nt0+nu):nu:ntf];

    for j = 1:nmcro
    
        fpde_split(:,1:nt0,j,k) = squeeze(fqoi(:,1:nt0,j));
        fg = zeros(nx+2*ng,1);     

        % Solve for defect/source term
        for i = (nt0+nu):nu:ntf
            fg(idx) = fqoi(:,i-nu,j);
            for ii = 1:nu
                ftmp = laxWen1d(fg, idx, nx, squeeze(advect(:,i-nu+ii,j)), dx, dt1);
                ftmp(ftmp<0) = eps;
                fg(idx) = ftmp;
            end
            
            if i == (nt0+nu)
                src(:,i,j,k) = (squeeze(fqoi(:,i,j)) - ftmp)/(nu*dt1);
                src(:,i-nu,j,k) = src(:,i,j,k);
            else
                src(:,i,j,k) = 2*(squeeze(fqoi(:,i,j)) - ftmp)/(nu*dt1) ...
                             - squeeze(src(:,i-nu,j,k));
            end
        end
         
        % Re-solve for solution
        for i = (nt0+nu):nu:ntf
            fg(idx) = squeeze(fpde_split(:,i-nu,j,k));
            for ii = 1:nu
                ftmp = laxWen1d(fg, idx, nx, squeeze(advect(:,i-nu+ii,j)), dx, dt1);
                ftmp(ftmp<0) = eps;
                fg(idx) = ftmp;
            end

            if i == (nt0+nu)
                ftmp = ftmp + (nu*dt1)*src(:,i,j,k);
            else
                ftmp = ftmp + 0.5*(nu*dt1)* (src(:,i,j,k) + src(:,i-nu,j,k));
            end
            ftmp(ftmp<0) = eps;
            fpde_split(:,i,j,k) = ftmp;
        end

        [T1, X1] = meshgrid(t1(idt), grid);
        [T2, X2] = meshgrid(t1(1:ntf), grid);
        fpde_split(:,:,j,k) = interp2(T1,X1,squeeze(fpde_split(:,idt,j,k)),...
                                      T2,X2,'makima');
        
    end
end


%% Kinectic Defect - Newtonian relaxation (a.k.a., nudging)

fpde_nr = zeros(nx,ntf,nmcro,nnu);     % PDF
Fpde_nr = zeros(nx,ntf,nmcro,nnu);     % CDF

l0 = 1.6/dt1;   % learning rate 

for k = 1:nnu
    nu = nu_vec(k);       % sparsity factor
    lr = zeros(ntf,1);   % learning rate
    lr([1:nt0,(nt0+nu):nu:ntf]) = nu*l0;
    for j = 1:nmcro
    
        fpde_nr(:,1:nt0,j,k) = fmc(:,1:nt0);
        fg = zeros(nx+2*ng,1);     
         
        for i = (nt0+1):ntf

            fg(idx) = squeeze(fpde_nr(:,i-1,j,k));
            ftmp = laxWen1d(fg, idx, nx, squeeze(advect(:,i,j)), dx, dt1);
           
            if mod(i-nt0,nu) == 0
                % df/dt = lr(t)*(fqoi - f)
                fpde_nr(:,i,j,k) = (ftmp + 0.5*dt1*( squeeze(lr(i)*fqoi(:,i,j)...
                                                   + lr(i-1)*fqoi(:,i-1,j)) ...
                                                   - lr(i-1)*ftmp ))...
                                 / (1 + 0.5*lr(i)*dt1);
            else
                fpde_nr(:,i,j,k) = ftmp;
            end

           % (f1 - f0)/dt1 = 0.5*(lam1*(fqoi1-f1) + lam0*(fqoi0 - f0))
           % lam0 = 0 when sparse

           % f1 = (f0  +  0.5*dt1*(lam1*fqoi1 + lam0*fqoi0 - lam0*f0)) ...
           %    / (1 + 0.5*lam1*dt1)

           Fpde_nr(:,i,j,k) = trapz(grid, squeeze(fpde_nr(:,i,j,k)));

        end
        
    end
end

%% Save

% save('ROPDF.mat','grid','nx','nu_vec','nnu','nt0','t1','nt1','ntf',...
%      'mcro_vec','nmcro','fmc','fqoi','cfls','fpde','fpde_split','src','fpde_nr',...
%      'l0','qoiMin','qoiMax','-v7.3')

% save('ROPDF2.mat','grid','nx','nu_vec','nnu','nt0','t1','nt1','ntf',...
%      'mcro_vec','nmcro','fmc','fqoi','cfls','fpde','fpde_split','src','fpde_nr',...
%      'l0','qoiMin','qoiMax','-v7.3')
