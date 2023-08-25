function ff = diffusion(f,f_ind,nx,D,dx,dt)

    % Takes one time step of 1d heat equation f_t = D*f_xx
    %   via Crank Nicolson and central differencing
    
    % Input:
    %   f := solution at current times step, (nx+2*ng,1) vector
    %   f_ind := indices of f of non-ghost cells
    %   nx := number of non-ghost cells
    %   D := scalar diffusion coefficient
    %   dt and dx are time and spatial step
    
    % Output: 
    %   f0 := (nx,1) solution at next time step on non-ghost cells
    
    dtdx2 = dt/(dx*dx);
    
    % Crank-Nicolson Matrix --> homogenous Dirichelt BCs
    oo = ones(nx,1);
    M = spdiags([-0.5*D*dtdx2*oo, (1+D*dtdx2)*oo, -0.5*D*dtdx2*oo],...
        -1:1,nx,nx);
    
    ff = M\(f(f_ind) + 0.5*D*dtdx2*(f(f_ind-1) - 2*f(f_ind) + f(f_ind+1)));
    
    if ~isequal(size(ff),[nx,1])
        error('ldiffusion output has incorrect size')
    end

end