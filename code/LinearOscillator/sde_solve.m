function u_sol = sde_solve(sde_scheme)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hongli Zhao, honglizhaobob@uchicago.edu
%          
% Simulates the full trajectory of the SDE given initial condition and 
% time grid (uniform spacing). 
%
%   Inputs:
%       sde_scheme                  Struct that defines the SDE stepping
%                                   scheme. It should have a member
%                                   function `.step(t, u)` which takes 
%                                   a single step of the SDE.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dimension of the problem
d = length(sde_scheme.u0);
% size of time grid
nt = length(sde_scheme.tgrid);

u_sol = zeros(d, nt);
u_sol(:, 1) = sde_scheme.u0;
for i = 2:nt
    u_sol(:,i) = sde_scheme.step(sde_scheme.tgrid(i-1), u_sol(:,i-1));
end
end