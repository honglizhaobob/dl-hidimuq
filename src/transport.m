function pnext = transport(p1,coeff1,coeff2,dx,dy,dt)
    % Corner transport scheme to take a time step of dt for the 2d 
    % advection equation of form:
    %   p_t + ( c1(x, y) * p )_x + ( c2(x, y) * p )_y = 0
    %
    % Ref: LeVeque sect 20.9

    pnext = p1;
    % get size of cell centers mesh
    [nx,ny] = size(p1(3:end-2,3:end-2));
    
    % positive and negative parts
    up = max(coeff1,0);
    um = min(coeff1,0);
    vp = max(coeff2,0);
    vm = min(coeff2,0);
    
    % left-right fluxes, i.e. x-direction

    % Apdq(i-1/2,j)
    Apdq = (up(3:end-2,3:end-2) + um(4:end-1,3:end-2)).*p1(3:end-2,3:end-2)...
            - (up(3:end-2,3:end-2).*p1(2:end-3,3:end-2) ...
            + um(3:end-2,3:end-2).*p1(3:end-2,3:end-2));
    % Amdq(i+1/2,j)
    Amdq = (up(4:end-1,3:end-2).*p1(3:end-2,3:end-2) ...
            + um(4:end-1,3:end-2).*p1(4:end-1,3:end-2))...
            - (up(3:end-2,3:end-2) + um(4:end-1,3:end-2)).*p1(3:end-2,3:end-2); 

    % up-down fluxes, i.e. y-direction
    Bpdq = (vp(3:end-2,3:end-2) + vm(3:end-2,4:end-1)).*p1(3:end-2,3:end-2)...
            - (vp(3:end-2,3:end-2).*p1(3:end-2,2:end-3) ...
            + vm(3:end-2,3:end-2).*p1(3:end-2,3:end-2));

    Bmdq = (vp(3:end-2,4:end-1).*p1(3:end-2,3:end-2) ...
            + vm(3:end-2,4:end-1).*p1(3:end-2,4:end-1))...
            - (vp(3:end-2,3:end-2) + vm(3:end-2,4:end-1)).*p1(3:end-2,3:end-2);
    
    
    % x-direction
  
    xWp = p1(4:end-1,2:end-1) - p1(3:end-2,2:end-1);
    xWm = p1(3:end-2,2:end-1) - p1(2:end-3,2:end-1);
    
    % y-direction
    yWp = p1(2:end-1,4:end-1) - p1(2:end-1,3:end-2);
    yWm = p1(2:end-1,3:end-2) - p1(2:end-1,2:end-3);
    
    % allocate thetas;
    
    % for Theta at i+1/2, j+1/2
    xThp = zeros(nx+4,ny+4); yThp = zeros(nx+4,ny+4);
    % for Theta at i-1/2, j-1/2
    xThm = zeros(nx+4,ny+4); yThm = zeros(nx+4,ny+4);
    
    % 1. Thetas for x direction
    % for theta x, loop over rows of u and find
    % > 0 and <= 0 entries, and fill in Theta 
    % correspondingly
    
    for r = 3:nx-2 % i-1/2,j => xThm 
       u_row = coeff1(r,:);
       % I = i-1
       positive_entries = find(u_row > 0);
       % I = i+1
       negative_entries = find(u_row <= 0);
       
       % these entries would correspond to theta entries...
       % note u is defined at i-1/2,j ...
       for idx = 1:length(positive_entries)
           % sect. (6.61) and (9.74)
           % W(I-1/2,j)/W(i-1/2,j) = W(i-3/2,j)/W(i-1/2,j)
           % W(i-3/2,j) = Q(i-1,j) - Q(i-2,j)
           c = positive_entries(idx);
           xThm(r,c) = (p1(r-1,c) - p1(r-2,c))/(p1(r,c) - p1(r-1,c));
       end
       for idx = 1:length(negative_entries)
           % W(I-1/2,j)/W(i-1/2,j) = W(i+1/2,j)/W(i-1/2,j)
           % W(i+1/2,j) = Q(i+1,j) - Q(i,j)
           c = negative_entries(idx);
           xThm(r,c) = (p1(r+1,c) - p1(r,c))/(p1(r,c) - p1(r-1,c));
       end
    end
    
    for r = 4:nx-1 % i+1/2,j => xThp
        u_row = coeff1(r,:);
        % I = i-1
        positive_entries = find(u_row > 0);
        % I = i+1
        negative_entries = find(u_row <= 0);
        
        % same logic, but for right cell edge thetas...
        for idx = 1:length(positive_entries)
            % W(I+1/2,j)/W(i+1/2,j) = W(i-1/2,j)/W(i+1/2,j)
            % W(i+1/2,j) = Q(i+1,j) - Q(i,j)
            c = positive_entries(idx);
            xThp(r,c) = (p1(r,c) - p1(r-1,c)) / (p1(r+1,c) - p1(r,c));
        end
        
        for idx = 1:length(negative_entries)
           % W(I+1/2,j)/W(i+1/2,j) = W(i+3/2,j)/W(i+1/2,j)
           % W(i+3/2,j) = Q(i+2,j) - Q(i+1,j)
           c = negative_entries(idx);
           xThp(r,c) = (p1(r+2,c) - p1(r+1,c)) / (p1(r+1,c) - p1(r,c));
        end
        
    end
    
    % 2. Thetas for y direction, similar logic
    
    for c = 3:ny-2 % i,j-1/2 => yThm
        v_col = coeff2(:,c);
        
        % I = i-1
        positive_entries = find(v_col > 0);
        % I = i+1
        negative_entries = find(v_col <= 0);
        
        for idx = 1:length(positive_entries)
            % W(i,J-1/2)/W(i,j-1/2) = W(i,j-3/2)/W(i,j-1/2)
            % W(i,j-3/2) = Q(i,j-1) - Q(i,j-2)
            r = positive_entries(idx);
            yThm(r,c) = (p1(r,c-1) - p1(r,c-2))/(p1(r,c) - p1(r,c-1));
        end
        
        for idx = 1:length(negative_entries)
            % W(i,J-1/2)/W(i,j-1/2) = W(i,j+1/2)/W(i,j-1/2)
            % W(i,j+1/2) = Q(i,j+1) - Q(i,j)
            r = negative_entries(idx);
            yThm(r,c) =  (p1(r,c+1) - p1(r,c))/(p1(r,c) - p1(r,c-1));
        end
    end
    
    for c = 4:ny-1 % i,j+1/2 => yThp
        v_col = coeff2(:,c);
        
        % I = i-1
        positive_entries = find(v_col > 0);
        % I = i+1
        negative_entries = find(v_col <= 0);
        
        for idx = 1:length(positive_entries)
           r = positive_entries(idx);
           % W(i,J+1/2)/W(i,j+1/2) = W(i,j-1/2)/W(i,j+1/2)
           % Q(i,j)-Q(i,j-1) / Q(i,j+1) - Q(i,j)
           yThp(r,c) = (p1(r,c) - p1(r,c-1)) / (p1(r,c+1) - p1(r,c));
        end
        
        for idx = 1:length(negative_entries)
           r = negative_entries(idx);
           % W(i,J+1/2)/W(i,j+1/2) = W(i,j+3/2)/W(i,j+1/2)
           % Q(i,j+2)-Q(i,j+1) / Q(i,j+1)-Q(i,j)
           yThp(r,c) = (p1(r,c+2) - p1(r,c+1)) / (p1(r,c+1) - p1(r,c));
        end
    end
   
    % 3. all thetas are filled in, do limiters (Van Leer)
    xPhim = max(max(0,min(1,2*xThm)),min(2,xThm));
    xPhip = max(max(0,min(1,2*xThp)),min(2,xThp));
    yPhim = max(max(0,min(1,2*yThm)),min(2,yThm));
    yPhip = max(max(0,min(1,2*yThp)),min(2,yThp));
    
    % modify waves 
    xWp = xPhip(3:end-2,2:end-1).*xWp;
    xWm = xPhim(3:end-2,2:end-1).*xWm;
    yWp = yPhip(2:end-1,3:end-2).*yWp;
    yWm = yPhim(2:end-1,3:end-2).*yWm;
    
    % second-order corrections
    % y-direction
    %(i,j+1/2)
    Gp = 0.5*abs(coeff2(2:end-1,4:end-1)).*(1-(dt/dx)* ...
        abs(coeff2(2:end-1,4:end-1))).*yWp;
    %(i,j-1/2)
    Gm = 0.5*abs(coeff2(2:end-1,3:end-2)).*(1-(dt/dx)* ...
        abs(coeff2(2:end-1,3:end-2))).*yWm;
    
    % x-direction
    %(i+1/2,j)
    Fp = 0.5*abs(coeff1(4:end-1,2:end-1)).*(1-(dt/dx)* ...
        abs(coeff1(4:end-1,2:end-1))).*xWp;
    %(i-1/2,j)
    Fm = 0.5*abs(coeff1(3:end-2,2:end-1)).*(1-(dt/dx)* ...
        abs(coeff1(3:end-2,2:end-1))).*xWm;

    % update solution to the next time step
    pnext(3:end-2,3:end-2) = p1(3:end-2,3:end-2) - dt*((Apdq + Amdq)/dx ...
        + (Bpdq + Bmdq)/dy + (Fp(:,2:end-1)-Fm(:,2:end-1))/dx ...
        + (Gp(2:end-1,:)-Gm(2:end-1,:))/dy);
end