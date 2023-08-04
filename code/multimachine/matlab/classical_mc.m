function paths_mc = classical_mc(mc,N,H,D,wr,Pm,g,b,v,R,T1,T2,u0,nt,dt,dt0,tf,sig)
    % Computes sample trajectories of classical multimachine power model 
    %   with Gaussian white noise via Euler's method

    % Input: 
    % mc  = (integer>0) # of MC sample trajectories.
    % N   = (integer>0) # of oscillators ==> 2*N states/equations in the system.
    % H   = (N-vector) interia coeffs.
    % D   = (N-vector) damping coeffs.
    % wr  = (scalar)
    % Pm  = (N-vector) bus power injections
    % g   = (NxN matrix) admittance matrix for cosine (generators)
    % b   = (NxN matrix) admittance matrix for sine (buses)
    % v   = (Nxmc matrix) random voltages
    % R   = (scalar) Droop
    % T1  = (scalar) transient gain time
    % T2  = (scalar) governor time constant
    % u0  = (3*Nxmc matrix) random initial conditions at time(1) 
    % nt  = (interger>0) number of coarse time nodes 
    % dt  = (scalar>0) coarse time step for PDE scheme 
    % dt0 = (scalar>0) refined time step for SDE scheme 
    % .      dt0 should divide dt.
    % tf  = (scalar>0) final time
    % sig = (scalar>0) variance for Gaussian noise
    

    % Output:
    %   paths_mc = 2N x nt x mc solution matrix. (i,j,k) is the k-th MC sample 
    %              trajectory of the i-th state at time(j)

   
    % Allocate solution
    paths_mc = zeros(3*N,nt,mc);  
         
    rat = dt/dt0;  % ratio of time steps
    parfor k = 1:mc 
        uk = u0(:,k); vk = v(:,k); 
        
        for n = 2:nt
             for n0 = 1:rat
                  uk = uk + dt0*rhs(uk,N,H,D,wr,Pm,g,b,vk,R,T1,T2);
                  uk(1:3:(3*N-2)) = uk(1:3:(3*N-2)) + sig.*sqrt(dt0).*randn(N,1);

             end
             paths_mc(:,n,k) = uk;
         end
    end
end


function uP = rhs(u,N,H,D,wr,Pm,g,b,vi,R,T1,T2)
    % computes vector-valued drift term of SDE
    uP = zeros(3*N,1);    
    umat = repmat(u(2:3:(3*N-1)),1,N);
    umat = umat - umat';
    
    % speed differential: w - wr
    sdiff = u(1:3:(3*N-2)) - wr;
    
    % power injection
    Tm = u(3:3:3*N) - T1*sdiff/(R*T2) + Pm;
    
    % speeds
    uP(1:3:(3*N-2)) = (Tm - D.*sdiff - vi.*sum(repmat(vi',N,1)...
        .*(g.*cos(umat) + b.*sin(umat)),2))*0.5*wr./H;
    
    % angles
    uP(2:3:(3*N-1)) = sdiff;

    % Governor: xg' = ((1/R)*(1-T1/T2)*(wr - w) - xg)/T2
    uP(3:3:3*N) = (-(1 - T1/T2)*sdiff/R - u(3:3:3*N))/T2;
    
end