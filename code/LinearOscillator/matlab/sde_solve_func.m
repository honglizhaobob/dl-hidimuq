function v_sol = sde_solve_func(sde_scheme, func, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hongli Zhao, honglizhaobob@uchicago.edu
%          
% 
%   A modification of `sde_solve.m` such that the simulated trajectory
%   is not stored, due to memory constraints. This function returns instead
%   a (Nt x 1) vector containing the values of any parameterized function
%   evaluated on the states at each time. 
%
%   Inputs:
%       sde_scheme                  Struct that defines the SDE stepping
%                                   scheme. It should have a member
%                                   function `.step(t, u)` which takes 
%                                   a single step of the SDE.
%       func                        a function handle that computes a
%                                   quantity of interest given the states.
%       
%       params                      Struct containing any other parameters
%                                   used in `func`.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% size of time grid
nt = length(sde_scheme.tgrid);
% preallocate
v_sol = zeros(nt,1);

% initialize and only keep track of the one-step vector
u_sol = sde_scheme.u0;
v_sol(1) = func(u_sol(:),params);
for i = 2:nt
    u_sol(:) = sde_scheme.step(sde_scheme.tgrid(i-1), u_sol(:));
    v_sol(i) = func(u_sol(:), params);
end
end