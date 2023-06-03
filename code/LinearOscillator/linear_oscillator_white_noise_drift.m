function mu = linear_oscillator_white_noise_drift(t, u, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hongli Zhao, honglizhaobob@uchicago.edu
%          
% Drift term for the linear oscillator driven by white noise.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = length(u);
assert(mod(n,2)==0);
mu = oscillator_rhs(t, u, params.M, params.D, params.K, params.f);
end