%     AUTHOR: Tyler E Maltba

%     AFFILIATIONS: 
%        Argonne National Lab: MCS Division
%                   Positions: Givens Associate, Research Aide
%                 Supervisors: Daniel Adrian Maldonado, Vishwas Rao
%
%        UC Berkeley: Dept. of Statistics
%          Positions: PhD Candidate, NSF Graduate Research Fellow

%     LAST UPDATE: Oct. 25, 2021
              
%     DESCRIPTION:
%        Generates all MC paths of classical multimachine power model with 
%        simple governor and correlated OU noise via Euler's method. Output
%        times are (dt0/2):dt0:((dt0/2) + nt0*dt)

function [u_path, xi_path] = classical_OU(mc,nt0,dt0,u0,N,H,D,wr,Pm,g,b,v,nxi,xi0,th,vol)

    % Input: 
    % mc   = (integer>0) # of MC sample trajectories.
    % nt   = (integer>0) # of total coarse time steps
    % rat  = (integer>0) # of intermediate time steps
    % dt0  = (scalar>0) refined time step for SDE scheme --> rat = dt/dt0
    % u0   = (2*Nxmc matrix) MC samples of state ICs
    % N    = (integer>0) # of oscillators/states (currently 3)
    % H    = (N-vector) interia coeffs.
    % D    = (N-vector) damping coeffs.
    % wr   = (N-vector) equilibrium speeds
    % Pm   = (N-vector) bus power injections
    % g    = (NxN matrix) admittance matrix for cosine (generators)
    % b    = (NxN matrix) admittance matrix for sine (buses)
    % v    = (Nxmc matrix) random voltages
    % sd_v = (scalar) st. dev. for random voltages
    % nxi  = (integer>0) # of noise processes
    % xi   = (nxi*Nxmc matrix) MC noise at current time
    % th   = (N-vector>0) drift constant for OU noise
    % vol  = (nxi*N x nxi*N matrix) Volatility matrix
    

    % Output:
    % u_path  = (2*N x nt x mc array) All mc paths of states 
    % xi_path = (nxi*N x nt x mc array) noise values at next coarse time step
    
    % Check inputs --> comment out if you know inputs are correct
    if ~(mc >= 1 && rem(mc,1)==0 && isscalar(mc) && isreal(mc))
        error('Need (real) integer mc > 0')
    end
    if ~(nt0 >= 1 && rem(nt0,1)==0 && isscalar(nt0) && isreal(nt0))
        error('Need (real) integer nt > 0')
    end
    if ~(isscalar(dt0) && isreal(dt0) && dt0>0)
        error('Need real scalar dt0 > 0 such that mod(dt,dt0)=0')
    end
    if ~(isequal(size(u0),[2*N,mc]) && isreal(u0))
        error('Need u0 a real 2*Nxmc matrix')
    end
    if ~(N >= 1 && rem(N,1)==0 && isscalar(N) && isreal(N))
        error('Need (real) integer N > 0')
    end
    if D < 0 | ~(isequal(size(D),[N,1]) && isreal(D)) %#ok<OR2>
        error('Need D >= 0 a real Nx1 vector')
    end
    if H <= 0 | ~(isequal(size(H),[N,1]) && isreal(H)) %#ok<OR2>
        error('Need H>0 a real Nx1 vector')
    end
    if ~(isequal(size(wr),[N,1]) && isreal(wr))
        error('Need real scalar wr')
    end
    if ~(isequal(size(Pm),[N,1]) && isreal(Pm))
        error('Need Pm a real vector')
    end
    if ~(isequal(size(g),[N,N]) && isreal(g))
        error('Need g a real NxN matrix')
    end
    if ~(isequal(size(b),[N,N]) && isreal(b))
        error('Need b a real NxN matrix')
    end
    if ~(isequal(size(v),[N,mc]) && isreal(v))
        error('Need Pm a real Nxmc matrix')
    end
    if ~(nxi >= 0 && rem(nxi,1)==0 && isscalar(nxi) && isreal(nxi))
        error('Need (real) integer nxi >= 0')
    end
    if ~(isequal(size(xi0),[nxi*N,mc]) && isreal(xi0))
        error('Need xi0 a real nxi*Nxmc matrix')
    end
    if th<=0 | ~(isequal(size(th),[nxi*N,1]) && isreal(th))  %#ok<OR2>
        error('Need th>0 a real nxi*Nx1 vector')
    end
    if ~(isequal(size(vol),[nxi*N,nxi*N]) && isreal(vol)) 
        error('Need C a real nxi*N x nxi*N matrix')
    end
   
    % Allocate
   u_path = zeros(2*N,nt0,mc);  xi_path = zeros(nxi*N,nt0,mc);
    
    % Main mc loop --> use regular for loop if parallel package not installed
    parfor k = 1:mc 
        
        % Random voltages (folded gaussian)
        vk = v(:,k);
        % states (uk)   % OU noise (xik):
        uk = u0(:,k);   xik = xi0(:,k);
        
        for n = 1:nt0
            if n==1
                dt = dt0/2;
            else
                dt = dt0;
            end
            
             % take time step 
              uk = uk + dt*rhs(uk,N,H,D,wr,Pm,g,b,vk);
              
             % add noise states
              if nxi >= 1   % noise for speeds
                uk(1:N) = uk(1:N) + dt*xik(1:N).*0.5.*wr./H;
                if nxi == 2 % noise for angles
                    uk((N+1):nxi*N) = uk((N+1):nxi*N) + dt*xik((N+1):nxi*N);
                end
                
                % Generate OU noise for next step:
                %   dxi = -th*xi*dt + vol*dW
                xik = xik - th.*xik*dt + sqrt(dt)*vol*randn(nxi*N,1);
                xi_path(:,n,k) = xik;
              end

             u_path(:,n,k) = uk;
         end
    end
end


function uP = rhs(uk,N,H,D,wr,Pm,g,b,vk)

    % computes vector-valued drift term of SDE
    uP = zeros(2*N,1);  
    % Matrix of angles
    umat = repmat(uk((N+1):2*N),1,N);
    umat = umat - umat';
    
    % speed differential: w - wr
    sdiff = uk(1:N) - wr;
    
    % speeds
    uP(1:N) = (Pm - D.*sdiff - vk.*sum(repmat(vk',N,1)...
        .*(g.*cos(umat) + b.*sin(umat)),2)).*0.5.*wr./H;
    
    % angles
    uP((N+1):2*N) = sdiff;  
end