function ff = lax_wen(f,f_ind,nx,u,dx,dt)

    % Takes one time step of 1d conservative advection equation
    %   via lax-wendroff with MC limiter
    
    % Reference: LeVeque, Randall J. Finite volume methods for hyperbolic problems.
    %   Vol. 31. Cambridge university press, 2002.
    
    % Input:
    %   f := solution at current times step, (nx+2*ng,1) vector
    %   f_ind := indices of f of non-ghost cells
    %   nx := number of non-ghost cells
    %   u := variable advec. coeff., (nx+1,1) vector 
    %        defined on left cell edges (1-1/2):(nx+1/2)
    %   dt and dx are time and spatial step
    
    % Output: 
    %   f0 := (nx,1) solution at next time step on non-ghost cells
    
    % Positive and negative speeds
    indp = find(u>0); indm = find(u<0);
    up = zeros(nx+1,1); um = up;
    up(indp) = u(indp); um(indm) = u(indm);
    
    % 1st-order right and left going flux differences
    % LeVeque sect. 9.5.2 The conservative equation

    % At cell i: Apdq(i-1/2) = right going  flux = F(i) - F(i-1/2),
    %            Amdq(i+1/2) = left going  flux  = F(i+1/2) - F(i),
    %            where F is numerical flux.
    % Upwind edge flux: F(i-1/2) = up(i-1/2)f(i-1) + um(i-1/2)f(i),
    %                   F(i-1/2) = up(i-1/2)f(i-1) + um(i-1/2)f(i).
    % Cell flux can be taken arbitrarily, i.e. F(i) = 0,
    %   but more asthetic to approximte it:
    %   F(i) = (up(i-1/2) + um(i+1/2)*f(i).
    
    % Apdq(i-1/2)= F(i) - F(i-1/2),  Amdq(i+1/2) = F(i+1/2) - F(i);
    % F(i-1/2) = up(i-1/2)f(i-1) + um(i-1/2)f(i)
    % F(i+1/2) = up(i+1/2)f(i) + um(i+1/2)f(i+1)
    
    % F(i)
%     Fi = (up(f_ind) + um(f_ind+1)).*f(f_ind);
    Flux_i = 0;
    % F(i-1/2)
    Flux_m = up(1:nx).*f(f_ind-1) + um(1:nx).*f(f_ind);
    % F(i+1/2)
    Flux_p = up(2:nx+1).*f(f_ind) + um(2:nx+1).*f(f_ind+1);
    % Apdq(i-1/2) and Amdq(i+1/2)
    Apdq = Flux_i - Flux_m;  Amdq = Flux_p - Flux_i;

    % W = wave with speed u; p = i+1/2, m = i-1/2
    Wp = f(f_ind+1) - f(f_ind); Wm = f(f_ind) - f(f_ind-1);

    % theta's for limiter: LeVeque book sect. 9.13
    % theta_i-1/2 = q(I) - q(I-1) / Wm , I = i-1 u_i-1/2>=0, =i+1 u_i-1/2<0
    % theta_i+1/2 = q(I+1) - q(I) / Wp , I = i-1 u_i+1/2>=0, =i+1 u_i+1/2<0
    
    % Allocate for limiters
    Thm =  zeros(nx,1); Thp = Thm;
    
    % At i-1/2
    xsm = indm(indm<nx+1); xsp = indp(indp<nx+1);
    Thm(xsm) = (f(f_ind(xsm)+1) - f(f_ind(xsm)))./Wm(xsm);     % negative speed
    Thm(xsp) = (f(f_ind(xsp)-1) - f(f_ind(xsp)-2))./Wm(xsp);   % positive speed
    
    % At i+1/2
    xsm = indm(indm>1)-1; xsp = indp(indp>1)-1;
    Thp(xsm) = (f(f_ind(xsm)+2) - f(f_ind(xsm)+1))./Wp(xsm);     % negative speed
    Thp(xsp) = (f(f_ind(xsp)) - f(f_ind(xsp)-1))./Wp(xsp);   % positive speed
    
    % MC limiter: LeVeque sect. 6.12 TVD Limiters eqn (6.39a)
    phip = max(0,min(min((1+Thp)/2,2),2*Thp));
    phim = max(0,min(min((1+Thm)/2,2),2*Thm));
    
    % mW = modified wave, LeVeque sect. 9.13 eqn (9.69)
    mWp = phip.*Wp; mWm = phim.*Wm;
      
    % 2nd-order limited corrections: LeVeque sect. 6.15 eqn (6.60)
    Fp = 0.5*abs(u(2:nx+1)).*(1 - (dt/dx)*abs(u(2:nx+1))).*mWp;
    Fm = 0.5*abs(u(1:nx)).*(1 - (dt/dx)*abs(u(1:nx))).*mWm;
    
    ff = f(f_ind) - (dt/dx)*(Apdq + Amdq + Fp - Fm);
    
    if ~isequal(size(ff),[nx,1])
        error('lax_wen output has incorrect size')
    end
end
