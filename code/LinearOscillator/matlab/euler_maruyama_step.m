function u_next = euler_maruyama_step(t, u, drift, diffusion, dt, eta, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hongli Zhao, honglizhaobob@uchicago.edu
% 
% Simulates 1 step of the Euler-Maruyama (EM) method, if the diffusion 
% term does not depend on the state, Milstein method coincides with EM.
%
%
%   Inputs:
%       eta                     Noise amplitude of the SDE.
%
%       params                  A struct containing additional parameters
%       that defines the underlying dynamical system.
%
%       drift                   A function handle that evaluates the drift
%       term, should be of format: Mu(t, Xt)
%
%       diffusion               A function handle that evaluates the
%       diffusion term, should be of format: D(t, Xt). For computational
%       efficiency, diffusion term includes the noise simulation.           
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% dimension of the problem
d = length(u);
% evaluate drift
adt = drift(t,u,params)*dt;
% evaluate diffusion
bdWt = diffusion(t,u,dt,eta,params);
% take a step of the SDE
u_next = u + adt + bdWt;
end