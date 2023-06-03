function DdWt = linear_oscillator_ou_noise_diffusion(t, u, dt, eta, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hongli Zhao, honglizhaobob@uchicago.edu
%          
% Diffusion term for the linear oscillator driven by Ornstein-Uhlenbeck 
% (OU) noise. This routine includes white noise term.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = length(u);
assert(mod(n,3)==0);
% state dimension
d = n/3;
% white noise
dWt = sqrt(dt).*randn(d,1);
DdWt = zeros(n,1);
% correlated white noise
DdWt((2*d+1):end) = eta.*(params.C*dWt);
end