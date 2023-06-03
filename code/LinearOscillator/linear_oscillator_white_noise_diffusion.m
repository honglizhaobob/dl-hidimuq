function DdWt = linear_oscillator_white_noise_diffusion(t, u, dt, eta, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hongli Zhao, honglizhaobob@uchicago.edu
%          
% Diffusion term for the linear oscillator driven by white noise, 
% including the white noise term.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = length(u);
assert(mod(n,2)==0);
% state dimension
d = n/2;
% white noise
dWt = sqrt(dt).*randn(d,1);
DdWt = zeros(n,1);
DdWt(d+1:end) = eta.*dWt;
end