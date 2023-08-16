%   pt  +  (u(x,y)p)x  +  (v(x,y)p)y  =  0

% test example for checking convergence of 
% main driver code corner_transport.m
% LeVeque sect 20.9, solid-body rotation
% 
% final time
tend = 3.1416;

% domain
x1 = -1;
x2 = -x1;
y1 = -1;
y2 = -y1;

% # of grid cells
nx = 200;
ny = nx;

% % grid edges
nX = nx+1;
X = linspace(x1,x2,nX);
nY = ny+1;
Y = X;

% % grid cells
dx = X(2)-X(1);
x = X(1:end-1) + dx/2;
dy = dx;
y = x;

% add 2 ghost cells

% cells
xg = [x(1)-2*dx, x(1)-dx, x, x(end)+dx, x(end)+2*dx];
yg = [y(1)-2*dy, y(1)-dy, y, y(end)+dy, y(end)+2*dy];
% left cell edges
xge = xg - dx/2;
yge = yg - dy/2;


% speeds defined at left and bottom cell edges
% cell edges
[Yge,Xge] = meshgrid(yge,xge);
% cell centers
[Yg,Xg] = meshgrid(yg,xg);

u = 2*Yg;
v = -2*Xg;


up = max(u,0);
um = min(u,0);
vp = max(v,0);
vm = min(v,0);

% CFL condtion and time step
dt = 1/(max(max(abs(u/dx)))+max(max(abs(v/dy))));
nt = ceil((tend/dt) + 1);
dt = tend/(nt-1);

% allocate
p = zeros(nx+4,ny+4);

% I.C.
I1 = find(Xg<.6 & Xg>.1 & Yg>-.25 & Yg<.25);
p(I1) = 1;
R = sqrt((Xg+.45).^2+Yg.^2);
I2 = find(R<.35);
p(I2) = 1-R(I2);

% figure(1)
% pcolor(Xg,Yg,p)
% shading interp

% choose limiter
disp('=> choice of limiters')
disp('==> 0 - no limiter')
disp('==> 1 - minmod')
disp('==> 2 - superbee')
disp('==> 3 - MC')
disp('==> 4 - van leer')
limiter_choice = input('=> please choose a limiter: 0 - 4:  ');
if limiter_choice == 1
   answer = input(['*> minmod currently gives you garbage,',...
                ' you sure you still want to see it? [0 - np/1 - yes]:  ']);
   if answer == 0
       ME = MException('MyComponent:noSuchVariable',...
                'bad choice, gotta pick again ');
       throw(ME);
   end
end
%%%%%%%%%%%%%%
% main time loop

for n = 1:nt

    t1 = dt*n;
    disp(t1);
    % Left-right fluxes, i.e. x-direction
    p1 = p;
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
    
    
    % waves - no limiter (ignore limiter for now)
    % x-direction
    
    xWp = p1(4:end-1,2:end-1) - p1(3:end-2,2:end-1);
    % xWp = p1(4:end-1,3:end-2) - p1(3:end-2,3:end-2);
    
    %size(xWp)
    xWm = p1(3:end-2,2:end-1) - p1(2:end-3,2:end-1);
    % xWm = p1(3:end-2,3:end-2) - p1(2:end-3,3:end-2);
    
    % y-direction
    yWp = p1(2:end-1,4:end-1) - p1(2:end-1,3:end-2);
    % yWp = p1(3:end-2,4:end-1) - p1(3:end-2,3:end-2);
    
    yWm = p1(2:end-1,3:end-2) - p1(2:end-1,2:end-3);
    % yWm = p1(3:end-2,3:end-2) - p1(3:end-2,2:end-3);
    
    % ========== add limiters - FIXME
    % LeVeque sect 6.12, high resolution limiters
    % find positions of pos/neg advection coeffs, and set I
    
    %xpositive = find(u(3:end-2,3:end-2) > 0);
    %xnegative = find(u(3:end-2,3:end-2) <= 0);
    %ypositive = find(v(3:end-2,3:end-2) > 0);
    %ynegative = find(v(3:end-2,3:end-2) <= 0);
    
    
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
       u_row = u(r,:);
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
        u_row = u(r,:);
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
        v_col = v(:,c);
        
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
        v_col = v(:,c);
        
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
    
    %%%%%%%%%
    %for i = 1:length(xpositive)
       % I = i-1
    %   curr_x = xpositive(i);
           % for x(i+1/2) => W(I+1/2) = W(i-1+1/2) = W(i-1/2)
       % xpWI = p1(3:end-2,2:end-1) - p1(2:end-3,2:end-1);

    %   xpWI = p1(curr_x+2, 2:end-1) - p1(curr_x+2 - 1, 2:end-1);
           % for x(i-1/2) => W(I-1/2) = W(i-1-1/2) = W(i-3/2)
       % xmWI = p1(2:end-3,2:end-1) - p1(1:end-4,2:end-1);
    %   xmWI = p1(curr_x+2 - 1, 2:end-1) - p1(curr_x+2 - 2, 2:end-1);
       %size(xpWI)
       %size(xWp(curr_x,:))
       %size(xThp)
    %   xThp(curr_x, :) = ( xpWI .* xWp(curr_x,2:end-1) ) ./...
    %       xWp(curr_x, 2:end-1).^2;
    %   xThm(curr_x, :) = ( xmWI .* xWm(curr_x,2:end-1) ) ./...
    %       xWm(curr_x, 2:end-1).^2;
    %end
    
    %for i = 1:length(xnegative)
    %   % I = i+1
    %   curr_x = xnegative(i);
    %      % for x(i+1/2) => W(I+1/2) = W(i+1+1/2) = W(i+3/2)
    %   xpWI = p1(curr_x + 1, 2:end-1) - p1(curr_x, 2:end-1);
    %      % for x(i-1/2) => W(I-1/2) = W(i+1-1/2) = W(i+1/2)
    %   xmWI = p1(curr_x, 2:end-1) - p1(curr_x - 1, 2:end-1);
    %   
    %   xThp(curr_x, :) = ( xpWI .* xWp(curr_x,2:end-1) ) ./...
    %       xWp(curr_x, 2:end-1).^2;
    %   xThm(curr_x, :) = ( xmWI .* xWm(curr_x,2:end-1) ) ./...
    %       xWm(curr_x, 2:end-1).^2;
    %end
    %%%%%%%%%%
    
    
    % 3. all thetas are filled in, do limiters
    
    switch limiter_choice
        case 0 
            disp('*> no limiter used')
            xPhip = 1; xPhim = 1;
            yPhip = 1; yPhim = 1;
        case 1
            disp('*> minmod')
            xPhim = (xThm + abs(xThm))./(1 + abs(xThm));
            xPhip = (xThp + abs(xThp))./(1 + abs(xThp));
            yPhim = (yThm + abs(yThm))./(1 + abs(yThm));
            yPhip = (yThp + abs(yThp))./(1 + abs(yThp));
        case 2
            disp('*> superbee')
            xPhim = max(0, min(min((1+xThm)/2,2), 2*xThm));
            xPhip = max(0, min(min((1+xThp)/2,2), 2*xThp));
            yPhim = max(0, min(min((1+yThm)/2,2), 2*yThm));
            yPhip = max(0, min(min((1+yThp)/2,2), 2*yThp));
        case 3
            disp('*> MC')
            xPhim = max(0, min(1, xThm));
            xPhip = max(0, min(1, xThp));
            yPhim = max(0, min(1, yThm));
            yPhip = max(0, min(1, yThp));
        case 4
            disp('*> van leer')
            xPhim = max(max(0,min(1,2*xThm)),min(2,xThm));
            xPhip = max(max(0,min(1,2*xThp)),min(2,xThp));
            yPhim = max(max(0,min(1,2*yThm)),min(2,yThm));
            yPhip = max(max(0,min(1,2*yThp)),min(2,yThp));
        otherwise
            % throws stacktrace
            ME = MException('MyComponent:noSuchVariable',...
                'Choice of limiter undefined! ');
            throw(ME);
    end
    
%     disp('==> size of xPhip and xWp')
%     size(xPhip(3:end-2,2:end-1))
%     size(xWp)
    % modify waves 
    if limiter_choice ~= 0
        xWp = xPhip(3:end-2,2:end-1).*xWp;
        xWm = xPhim(3:end-2,2:end-1).*xWm;
        yWp = yPhip(2:end-1,3:end-2).*yWp;
        yWm = yPhim(2:end-1,3:end-2).*yWm;
    end
    % ==========
    
    % second-order corrections
    % y-direction
    %(i,j+1/2)
    Gp = 0.5*abs(v(2:end-1,4:end-1)).*(1 - (dt/dx)*abs(v(2:end-1,4:end-1))).*yWp;
    %(i,j-1/2)
    Gm = 0.5*abs(v(2:end-1,3:end-2)).*(1 - (dt/dx)*abs(v(2:end-1,3:end-2))).*yWm;
    
    
%     Gp(2:end-1,:) = Gp(2:end-1,:) - 0.5*dt*v(3:end-2,4:end-1).*Apdq/dx;
%     Gp(1:end-2,:) = Gp(1:end-2,:) - 0.5*dt*v(2:end-3,4:end-1).*Apdq/dx;
    
    % Amdq(i-1/2,j)
%     Amdq2 = (up(3:end-2,3:end-2).*p1(2:end-3,3:end-2) ...
%             + um(3:end-2,3:end-2).*p1(3:end-2,3:end-2))...
%             - (up(2:end-3,3:end-2) + um(3:end-2,3:end-2)).*p1(2:end-3,3:end-2);
    
%     Gm(2:end-1,:) = Gm(2:end-1,:) - 0.5*dt*v(3:end-2,3:end-2).*Amdq2/dx;
%     Gm(1:end-2,:) = Gm(1:end-2,:) - 0.5*dt*v(2:end-3,3:end-2).*Amdq2/dx;
    
    % x-direction
    %(i+1/2,j)
    Fp = 0.5*abs(u(4:end-1,2:end-1)).*(1 - (dt/dx)*abs(u(4:end-1,2:end-1))).*xWp;
    %(i-1/2,j)
    Fm = 0.5*abs(u(3:end-2,2:end-1)).*(1 - (dt/dx)*abs(u(3:end-2,2:end-1))).*xWm;

%     Fp(:,2:end-1) = Fp(:,2:end-1) - 0.5*dt*u(4:end-1,3:end-2).*Bpdq/dy;
%     Fp(:,1:end-2) = Fp(:,1:end-2) - 0.5*dt*u(4:end-1,2:end-3).*Bpdq/dy;
    
    % Bmdq(i-1/2,j)
%     Bmdq2 = (vp(3:end-2,3:end-2).*p1(3:end-2,2:end-3) ...
%             + vm(3:end-2,3:end-2).*p1(3:end-2,3:end-2))...
%             - (vp(3:end-2,2:end-3) + vm(3:end-2,3:end-2)).*p1(3:end-2,2:end-3);
    
%     Fm(:,2:end-1) = Fm(:,2:end-1) - 0.5*dt*u(3:end-2,3:end-2).*Bmdq2/dy;
%     Fm(:,1:end-2) = Fm(:,1:end-2) - 0.5*dt*u(3:end-2,2:end-3).*Bmdq2/dy;
    
    %  continue section 21.2. I have done points 1-3. Need to do steps 4-7
    % ... once F and G are fully updated then
    
    p(3:end-2,3:end-2) = p1(3:end-2,3:end-2) - dt*((Apdq + Amdq)/dx ...
        + (Bpdq + Bmdq)/dy + (Fp(:,2:end-1)-Fm(:,2:end-1))/dx ...
        + (Gp(2:end-1,:)-Gm(2:end-1,:))/dy);
    
    figure(1)
    pcolor(Xg,Yg,p(:,:))
    shading interp
end